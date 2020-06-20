import sys, torch
from tqdm import tqdm
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import os.path as osp
import scipy.io as sio

from reid_models import *
from project_utils import *
from my_utils import *
from losses import *
from scheduler import create_scheduler
# Add eval.sysu to sys.path
current_path = osp.split(osp.abspath(__file__))[0]
sysu_evalscript_path = osp.join(current_path, "./eval_sysu")
sys.path += [sysu_evalscript_path]
from evaluate_SYSU_MM01 import evaluate_results

set_random_seed(0)


def get_model(attribute_choices, label2index):

    in_channels = 3
    model = create_reid_model(
        model_name=opt.arch,
        attribute_list=attribute_choices,
        in_chans=in_channels,
        num_classes=len(label2index),
        attr_hidden_size=None,
        pretrained=True,
    )

    return model


def AttributeCriterion(attribute_list, loss_criterion):
    attribute_criterion = {}
    for atrribute_name, choices in attribute_list.items():
        attribute_criterion[atrribute_name] = loss_criterion(len(choices), use_gpu=opt.cuda)
    return attribute_criterion


def calculate_attribute_loss(
    attribute_criterion, common_logits, target_attributes, attributes2index,
):
    attribute_classify_loss = 0
    for key, logit in common_logits.items():
        if (
            key != "id" and key != "upcloth" and key != "downcloth" and key != "domain"
        ):  # id based loss is handled in compute_person_loss, upcloth, downcloth are not used for IR, RGB reid
            current_attribute_target = target_attributes[:, attributes2index[key]]

            current_attribute_loss = attribute_criterion[key](
                logit, current_attribute_target.long()
            )

            # print(current_attribute_loss.shape)
            attribute_classify_loss += current_attribute_loss
            # print(attribute_classify_loss.shape)

    # handle the averaging per person
    relevant_attributes_count = (target_attributes != -1).sum(dim=1)

    # take average of the attributes loss for the persons with atleast one attribute
    # incase of PCB, there is no attribute loss implemented
    if not (opt.arch.startswith("pcb") or opt.arch.startswith("senet")):
        attribute_classify_loss = (
            attribute_classify_loss / (relevant_attributes_count.float() + 1e-6)
        ).mean()
    else:
        attribute_classify_loss = torch.tensor(0.0)

    return attribute_classify_loss


def calculate_triplet_loss(triplet_criterion, fc_features, target_person_labels):
    return triplet_criterion(fc_features, target_person_labels)


def calculate_attribute_triplet_loss(
    triplet_criterion, attribute_cat_features, target_person_labels
):
    return triplet_criterion(attribute_cat_features, target_person_labels)


def compute_person_loss(person_id_criterion, logit, target_person_labels):

    if isinstance(logit, (list, tuple)):
        person_classify_loss = DeepSupervision(
            person_id_criterion, logit, target_person_labels.long()
        )
    else:
        person_classify_loss = person_id_criterion(logit, target_person_labels.long())

    return person_classify_loss


def compute_domain_loss(domain_criterion, logit, target_labels):

    domain_classify_loss = domain_criterion(logit, target_labels.long())

    return domain_classify_loss


def train_epoch(
    model,
    dataLoader,
    optimizer,
    person_id_criterion,
    triplet_criterion,
    attribute_criterion,
    attributes2index,
    attribute_choices,
    epoch,
    total_epochs,
):

    # loss logistics
    total_lossmeter = AverageMeter()
    classify_person_lossmeter = AverageMeter()
    triplet_lossmeter = AverageMeter()
    attribute_lossmeter = AverageMeter()
    domain_lossmeter = AverageMeter()

    # Accuracy logistics
    person_accuracy = AccuracyMeter()
    attribute_accuracy = AttributeAccuracyMeter(
        attribute_choices, attributes2index, len(attribute_choices)
    )

    # Set modes for the models
    model.train()

    for batch_index, contents in enumerate(tqdm(dataLoader)):

        imgs, target_person_labels, target_attributes, rgb_or_IR = contents

        # Alpha factor for reverse gradient
        p = (
            float(batch_index + epoch * len(dataLoader))
            / (total_epochs - epoch)
            / len(dataLoader)
        )
        alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1

        # Convert to cuda tensor
        if opt.cuda:
            imgs = imgs.float().cuda()
            rgb_or_IR = rgb_or_IR.float().cuda()
            target_person_labels = target_person_labels.float().cuda()
            target_attributes = target_attributes.float().cuda()

        # Get the flags for the current batch
        rgb_flag = rgb_or_IR == category_label["rgb"]
        # zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        features, logits = model(imgs, alpha=alpha)

        # Compute loss for common logits
        person_loss = compute_person_loss(
            person_id_criterion, logits["id"], target_person_labels.long()
        )

        # Compute domain loss and add it to the total loss
        domain_loss = compute_domain_loss(
            person_id_criterion, logits["domain"], rgb_flag.long()
        )

        triplet_loss = calculate_triplet_loss(
            triplet_criterion, features, target_person_labels
        )
        # Attribute losses
        attribute_loss = calculate_attribute_loss(
            attribute_criterion, logits, target_attributes, attributes2index,
        )

        # multiply losses by respective lambdas
        person_classify_loss = opt.lambda_person_id * person_loss
        triplet_loss = opt.lambda_triplet * triplet_loss
        attribute_loss = opt.lambda_attribute * attribute_loss
        domain_loss = opt.lambda_domain * domain_loss
        total_loss = person_classify_loss + triplet_loss + attribute_loss + domain_loss

        # Backward + optimize
        total_loss.backward()
        optimizer.step()

        num_imgs = imgs.shape[0]
        # loss logistics update
        person_accuracy.update(logits["id"], target_person_labels.long())
        attribute_accuracy.update(
            logits,
            target_attributes,
            exclusion_list=["id", "upcloth", "downcloth", "domain"],
        )
        total_lossmeter.update(total_loss.item(), num_imgs)
        classify_person_lossmeter.update(person_classify_loss.item(), num_imgs)
        triplet_lossmeter.update(triplet_loss.item(), num_imgs)
        attribute_lossmeter.update(attribute_loss.item(), num_imgs)
        
        domain_lossmeter.update(domain_loss.item(), num_imgs)
        if (batch_index + 1) % opt.print_freq == 0:
            print(
                "epoch {} [{}/{}]: total avg {:.6f}, pers.classif. avg {:.6f}, triplet avg {:.6f}, attribute avg {:.6f}, domain avg {:.6f}".format(
                    epoch,
                    batch_index + 1,
                    len(dataLoader),
                    total_lossmeter.avg,
                    classify_person_lossmeter.avg,
                    triplet_lossmeter.avg,
                    attribute_lossmeter.avg,
                    domain_lossmeter.avg,
                )
            )

    print(
        "epoch {} [{}/{}]: total avg {:.6f}, pers.classif. avg {:.6f}, triplet avg {:.6f}, attribute avg {:.6f}, domain avg {:.6f}, person accuracy {:.6f}, attribute accuracy {:.6f}".format(
            epoch,
            batch_index + 1,
            len(dataLoader),
            total_lossmeter.avg,
            classify_person_lossmeter.avg,
            triplet_lossmeter.avg,
            attribute_lossmeter.avg,
            domain_lossmeter.avg,
            person_accuracy.accuracy(),
            attribute_accuracy.accuracy(),
        )
    )


def test(epoch, model, test_dataset, test_ids):
    def get_max_test_id(test_ids):
        int_test_ids = [int(ID) for ID in test_ids]
        return np.max(int_test_ids)

    def prepare_empty_matfile_config(max_test_id):
        cam_features = np.empty(max_test_id, dtype=object)
        for i in range(len(cam_features)):
            cam_features[i] = []
        return cam_features

    data_instances = test_dataset.get_cam_files_config()
    # print(len(data_instances), data_instances[0])
    matfile_prefix = "epoch_" + str(epoch)
    testresults_dir = os.path.join(opt.save_root, matfile_prefix)
    if not os.path.exists(testresults_dir):
        os.mkdir(testresults_dir)

    max_test_id = get_max_test_id(test_ids)

    model.eval()

    if opt.cuda:
        model.cuda()

    for cam_name, id_contents in data_instances.items():
        # data_instances
        #  --cam1
        #      -- 0001
        #           -- 0001.jpg
        #           -- 0002.jpg
        #           -- ...
        matfile_path = os.path.join(
            testresults_dir, matfile_prefix + "_" + cam_name + ".mat"
        )
        print(cam_name)

        # prepare empty features for all the person ids upto max_test_id
        # in the feature_original, features wrt all the images from all ids are extracted regardless
        # of whether it is a test subject or not
        # but this script only extracts the features for test subjects, that too only upto max_test_id (cam3 contains 533 ids, but 333 is the max test id)
        # other ids within 333 will have empty features (shape = (0,0))
        cam_features = prepare_empty_matfile_config(max_test_id)

        for id_, img_contents in tqdm(id_contents.items()):
            all_current_id_features = np.empty(shape=[0, model.feature_dim])
            for img_config in img_contents:
                # print(img_config)
                img, category = test_dataset.read_image_from_config(img_config)

                if opt.cuda:
                    img = img.unsqueeze(0).float().cuda()
                else:
                    img = img.unsqueeze(0)

                with torch.no_grad():
                    features, logits = model(img)
                    # features, logits, attr_features, attribute_cat_features = model(img)

                current_feature = features.data[0].cpu().numpy().reshape(1, -1)
                all_current_id_features = np.append(
                    all_current_id_features, current_feature, axis=0
                )

            cam_features[int(id_) - 1] = all_current_id_features
            # print(cam_features[int(id_)-1].shape)

        sio.savemat(matfile_path, {"feature": cam_features})

    if opt.test_on_val:
        test_mat_name = "val"
    else:
        test_mat_name = "test"

    evaluate_results(
        testresults_dir, matfile_prefix, opt.test_mode, opt.test_number_shot, test_mat_name = test_mat_name
    )


if __name__ == "__main__":
    # Read arguments
    opt = OptionsParser()

    # init the log
    determine_save_path(opt)
    sys.stdout = init_logger(opt)
    print(opt)

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # Get data from all three datasets
    (
        train_dataset,
        test_dataset,
        test_ids,
        attribute_list,
        attribute_choices,
        label2index,
        attributes2index,
    ) = get_datasets(opt)

    # category label
    category_label = train_dataset.category_type
    num_classes = len(label2index)
    print("Length of dataset = ", len(train_dataset))
    print("Number of classes = ", num_classes)

    sampler = RandomIdentity_IRRGBSampler(
        train_dataset.data_instances,
        batch_size=opt.train_batch,
        num_per_domain_instances=opt.num_instances,
    )

    dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.train_batch, sampler=sampler, num_workers=8,
    )

    # Get all the required models
    model = get_model(attribute_choices, label2index)

    # Use CUDA
    if opt.cuda:
        model.cuda()

    # load pretrained model if given
    epoch = 0
    if opt.pretrained_model != "":
        # load_pretrained_weights(model, opt.pretrained_model + "/model.pth.tar-" + str(opt.preepoch))
        load_pretrained_weights(model, opt.pretrained_model)
        # determine the epoch of pretrained model
        epoch = opt.preepoch

    # if only evaluation is required
    if opt.evaluate:
        print("-- evaluate only")
        test(epoch + 1, model, test_dataset, test_ids)
        exit(0)

    # Define optimizer and loss function
    person_id_criterion = CrossEntropyLabelSmooth(num_classes, use_gpu=opt.cuda)
    attribute_criterion = AttributeCriterion(attribute_choices, CrossEntropyLabelSmooth)
    triplet_criterion = TripletLoss(opt.margin, opt.gamma_triplet)
    print("Person ID loss = ", person_id_criterion)
    print("Attribute loss = ", attribute_criterion)
    print("Triplet loss = ", triplet_criterion)
    print("with gamma = ", opt.gamma_triplet)

    if opt.optim == "sgd":
        optimizer = optim.SGD(
            model.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4
        )  #
    else:
        optimizer = optim.Adam(
            model.parameters(), lr=opt.lr, weight_decay=5e-4
        )  # Default lr = 3e-4
    print("Optimizer = ", optimizer)

    num_epochs = opt.epochs
    # scheduler creation
    lr_scheduler, num_epochs = create_scheduler(opt, optimizer)

    if epoch > 0:
        lr_scheduler.step(epoch)
    print("Scheduled epochs: ", num_epochs)
    print(
        "learning rates ", [lr_scheduler._get_lr(epoch) for epoch in range(num_epochs)],
    )

    # Training routine
    while epoch < num_epochs:
        train_epoch(
            model,
            dataloader,
            optimizer,
            person_id_criterion,
            triplet_criterion,
            attribute_criterion,
            attributes2index,
            attribute_choices,
            epoch,
            num_epochs,
        )

        if epoch % opt.test_freq == 0 or (epoch == num_epochs - 1):
            test(epoch + 1, model, test_dataset, test_ids)

        lr_scheduler.step(epoch)

        # save model, if needed
        if epoch % opt.save_freq == 0 or (epoch == num_epochs - 1):
            save_checkpoint(
                {"state_dict": model.state_dict(), "epoch": epoch + 1},
                opt.save_root,
                m_name="model.pth.tar-",
            )

        epoch += 1
