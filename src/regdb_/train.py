import sys
import torch
from tqdm import tqdm
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import os.path as osp
import numpy as np
from scipy.spatial.distance import cdist

from regDB_project_utils import (
    init_logger_,
    get_dataset,
    OptionsParser,
    determine_save_path,
)
from regDB_project_utils import (
    AverageMeter,
    load_pretrained_weights,
    AccuracyMeter,
    AttributeAccuracyMeter,
)
from scheduler import create_scheduler
from reid_models import create_reid_model, Identity
from losses import TripletLoss, CrossEntropyLabelSmooth
from eval_regdb import *
from my_utils.torchtools import save_checkpoint
from dataset_regdb import Dataset, RandomIdentity_IRRGBSampler
from re_ranking_feature import re_ranking

import os
import sys
import humanize
import psutil
import GPUtil


def mem_report():
    print("CPU RAM Free: " + humanize.naturalsize(psutil.virtual_memory().available))
    GPUs = GPUtil.getGPUs()
    for i, gpu in enumerate(GPUs):
        print('GPU {:d} ... Mem Free: {:.0f}MB / {:.0f}MB | Utilization {:3.0f}%'.format(
            i, gpu.memoryFree, gpu.memoryTotal, gpu.memoryUtil*100))


def get_models(num_classes, attribute_choices):

    print("number of classes = ", num_classes)
    in_channels = 3
    model = create_reid_model(
        model_name=opt.arch, in_chans=in_channels, num_classes=num_classes,
        attribute_list=attribute_choices,
        attr_hidden_size=None,
        pretrained=True,
    )
    return model


def AttributeCriterion(attribute_list, loss_criterion):
    attribute_criterion = {}
    for atrribute_name, choices in attribute_list.items():
        attribute_criterion[atrribute_name] = loss_criterion(
            len(choices), use_gpu=opt.cuda)
    return attribute_criterion


def calculate_attribute_loss(
    attribute_criterion, common_logits, target_attributes, attributes2index,
):
    attribute_classify_loss = 0
    # Attribute logits are not present
    if len(common_logits) <= 2:
        return torch.tensor(0.0)

    for key, logit in common_logits.items():
        if (key != "id" and key != "domain"):
            # id based loss and domain loss are handled separately
            current_attribute_target = target_attributes[:,
                                                         attributes2index[key]]

            current_attribute_loss = attribute_criterion[key](
                logit, current_attribute_target.long()
            )

            # print(current_attribute_loss.shape)
            attribute_classify_loss += current_attribute_loss
            # print(attribute_classify_loss.shape)

    # handle the averaging per person
    relevant_attributes_count = (target_attributes != -1).sum(dim=1)
    attribute_classify_loss = (
        attribute_classify_loss / (relevant_attributes_count.float() + 1e-6)
    ).mean()

    return attribute_classify_loss


def calculate_classification_loss(person_id_criterion, logit, target_person_labels):
    return person_id_criterion(logit, target_person_labels.long())


def calculate_triplet_loss(triplet_criterion, fc_features, target_person_labels):
    return triplet_criterion(fc_features, target_person_labels)


def compute_domain_loss(domain_criterion, logit, target_labels):
    # Domain logits are not present
    if logit is None:
        return torch.tensor(0.0)
    return domain_criterion(logit, target_labels.long())


def train_epoch(
    model,
    dataloader,
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

    # Set modes for the model
    model.train()

    # Generate the person ID meter
    id_meter = AccuracyMeter()
    # Attribute accuracy meter
    disable_meter = False
    if opt.arch == "osnet_ain":
        disable_meter = True
    attribute_accuracy = AttributeAccuracyMeter(
        attribute_choices, attributes2index, len(attribute_choices), disable_meter=disable_meter,
    )

    for batch_index, contents in enumerate(tqdm(dataloader)):

        imgs, target_person_labels, target_attributes, v_or_t = contents

        # Alpha factor for reverse gradient
        p = (
            float(batch_index + epoch * len(dataloader))
            / (total_epochs - epoch)
            / len(dataloader)
        )
        alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1

        # Convert to cuda tensor
        if opt.cuda:
            imgs = imgs.float().cuda()
            v_or_t = v_or_t.float().cuda()
            target_person_labels = target_person_labels.float().cuda()
            target_attributes = target_attributes.float().cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # select visible images and thermal images
        v_flag = v_or_t == category_label["rgb"]

        # Forward pass
        fc_features, classifier_logits = model(imgs, alpha)

        # Compute loss
        person_classify_loss = calculate_classification_loss(
            person_id_criterion, classifier_logits["id"], target_person_labels.long(
            )
        )
        triplet_loss = calculate_triplet_loss(
            triplet_criterion, fc_features, target_person_labels
        )
        domain_loss = compute_domain_loss(
            person_id_criterion, classifier_logits["domain"], v_flag.long()
        )
        # Attribute losses
        attribute_loss = calculate_attribute_loss(
            attribute_criterion, classifier_logits, target_attributes, attributes2index,
        )

        # multiply losses by respective lambdas
        person_classify_loss = opt.lambda_person_id * person_classify_loss
        triplet_loss = opt.lambda_triplet * triplet_loss
        domain_loss = opt.lambda_domain * domain_loss
        attribute_loss = opt.lambda_attribute * attribute_loss

        total_loss = person_classify_loss + triplet_loss + domain_loss + attribute_loss

        # Backward + optimize
        total_loss.backward()
        optimizer.step()

        num_imgs = imgs.shape[0]
        # loss logistics update
        total_lossmeter.update(total_loss.item(), num_imgs)
        classify_person_lossmeter.update(person_classify_loss.item(), num_imgs)
        triplet_lossmeter.update(triplet_loss.item(), num_imgs)
        domain_lossmeter.update(domain_loss.item(), num_imgs)
        attribute_lossmeter.update(attribute_loss.item(), num_imgs)

        # accuracy meters
        attribute_accuracy.update(
            classifier_logits,
            target_attributes,
            exclusion_list=["id", "domain"],
        )
        id_meter.update(classifier_logits["id"], target_person_labels.long())

        if (batch_index + 1) % opt.print_freq == 0:
            print(
                "epoch {} [{}/{}]: total avg {:.6f}, pers.classif. avg {:.6f}, triplet avg {:.6f}, domain avg {:.6f}, attribute avg {:.6f}".format(
                    epoch,
                    batch_index + 1,
                    len(dataloader),
                    total_lossmeter.avg,
                    classify_person_lossmeter.avg,
                    triplet_lossmeter.avg,
                    domain_lossmeter.avg,
                    attribute_lossmeter.avg,
                )
            )

    print(
        "epoch {} [{}/{}]: total avg {:.6f}, pers.classif. avg {:.6f}, triplet avg {:.6f},  domain avg {:.6f}, person accuracy {:.6f}, attribute accuracy {:.6f}".format(
            epoch,
            batch_index + 1,
            len(dataloader),
            total_lossmeter.avg,
            classify_person_lossmeter.avg,
            triplet_lossmeter.avg,
            domain_lossmeter.avg,
            id_meter.accuracy(),
            attribute_accuracy.accuracy(),
        )
    )


def get_features(test_dataset, model):

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=opt.test_batch, shuffle=False
    )

    for batch_index, contents in enumerate(tqdm(test_dataloader)):

        # Read the contents
        imgs, target_person_labels, v_or_t = contents

        # Convert to cuda tensor
        if opt.cuda:
            imgs = imgs.float().cuda()

        # select visible images and thermal images
        v_flag = v_or_t == category_label["rgb"]
        t_flag = v_or_t == category_label["IR"]

        with torch.no_grad():
            fc_features, _ = model(imgs)

        # Move features to cpu
        fc_features = fc_features.cpu()

        # Query images are visible images and gallery images are thermal images
        if batch_index == 0:
            visible_features = fc_features[v_flag]
            visible_ids = target_person_labels[v_flag].long()
            thermal_features = fc_features[t_flag]
            thermal_ids = target_person_labels[t_flag].long()

        else:
            visible_features = torch.cat(
                (visible_features, fc_features[v_flag]))
            visible_ids = torch.cat(
                (visible_ids, target_person_labels[v_flag].long()))

            thermal_features = torch.cat(
                (thermal_features, fc_features[t_flag]))
            thermal_ids = torch.cat(
                (thermal_ids, target_person_labels[t_flag].long()))

    if opt.t_to_v:
        return thermal_features, thermal_ids, visible_features, visible_ids
    else:
        return visible_features, visible_ids, thermal_features, thermal_ids


def test(epoch, model):

    print("Testing model at epoch = " + str(epoch))
    # Set test modes for the models
    model.eval()
    if opt.cuda:
        model.cuda()

    # Trial number
    test_trial = opt.trial

    # Load test dataset
    test_dataset = Dataset(opt.dataroot, "test", test_trial)

    # Get the features of query and gallery instances
    gallery_features, gallery_ids, query_feataures, query_ids = get_features(
        test_dataset, model
    )

    # Use reranking if needed
    if opt.rerank:
        distmat = re_ranking(query_feataures, gallery_features,
                             k1=opt.k1, k2=opt.k2, lambda_value=opt.rerank_lambda)
    else:
        distmat = cdist(query_feataures, gallery_features)

    cmc, mAP, mINP = eval_regdb(
        distmat, query_ids.numpy(), gallery_ids.numpy())

    print("Test Trial: {}".format(test_trial))
    print(
        "Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}".format(
            cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP
        )
    )


if __name__ == "__main__":
    # Read arguments
    opt = OptionsParser()

    # init the log
    determine_save_path(opt)
    sys.stdout = init_logger_(opt)

    # Print out the options
    print(opt)

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # Get train and test datasets
    train_dataset = get_dataset(opt)
    # Get attribute related objects from train_dataset
    attribute_list = train_dataset.all_ids_attributes.attribute_list
    attribute_choices = train_dataset.all_ids_attributes.num_options_per_attributes
    attribute2index = train_dataset.all_ids_attributes.attribute2index

    # category label
    # visible = 1, thermal = 0
    category_label = train_dataset.category_type

    # Sampler defintion
    sampler = RandomIdentity_IRRGBSampler(
        train_dataset.data_instances,
        batch_size=opt.train_batch,
        num_per_domain_instances=opt.num_instances,
    )

    dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.train_batch, sampler=sampler, num_workers=1
    )

    # Get the model
    model = get_models(train_dataset.number_classes(), attribute_choices)

    # Use CUDA
    if opt.cuda:
        model.cuda()

    # load pretrained model if given
    epoch = 0
    if opt.pretrained_model != "":
        load_pretrained_weights(model, opt.pretrained_model)

        # determine the epoch of pretrained model
        epoch = int((opt.pretrained_model.split("/")[-1]).split("-")[-1])

    # if only evaluation is required
    if opt.evaluate:
        print("-- evaluate only")
        test(epoch, model)
        exit(0)

    # Define optimizer and loss function
    person_id_criterion = CrossEntropyLabelSmooth(
        train_dataset.number_classes(), use_gpu=opt.cuda
    )
    attribute_criterion = AttributeCriterion(
        attribute_choices, CrossEntropyLabelSmooth)
    triplet_criterion = TripletLoss(opt.margin)
    optimizer = optim.Adam(
        model.parameters(), lr=opt.lr, weight_decay=5e-4
    )  # Default lr = 3e-4

    print("Using triplet loss = ", triplet_criterion)
    print("Using person_id = ", person_id_criterion)
    print("Using Attribute loss = ", attribute_criterion)
    print("Optimizer = ", optimizer)

    # scheduler creation
    lr_scheduler, num_epochs = create_scheduler(opt, optimizer)

    if epoch > 0:
        lr_scheduler.step(epoch)
    print("Scheduled epochs: ", num_epochs)
    print(
        "learning rates ", [lr_scheduler._get_lr(
            epoch) for epoch in range(num_epochs)]
    )

    # Training routine
    while epoch < num_epochs:

        # Training procedure
        train_epoch(
            model,
            dataloader,
            optimizer,
            person_id_criterion,
            triplet_criterion,
            attribute_criterion,
            attribute2index,
            attribute_choices,
            epoch,
            num_epochs,
        )

        # Testing if needed
        if ((epoch) % opt.test_freq == 0) or epoch == 0 or (epoch == num_epochs - 1):
            test(epoch, model)
            mem_report()

        lr_scheduler.step(epoch)

        # save model, if needed
        if epoch % opt.save_freq == 0 or (epoch == num_epochs - 1):
            save_checkpoint(
                {"state_dict": model.state_dict(), "epoch": epoch + 1},
                opt.save_root,
                m_name="model.pth.tar-",
            )

        epoch += 1
