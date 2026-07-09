#!/usr/bin/python
# -*- encoding: utf-8 -*-


import lib.data.transform_cv2 as T
from lib.data.base_dataset import BaseDataset


class CustomerDataset(BaseDataset):

    def __init__(self, dataroot, annpath, trans_func=None, mode='train'):
        super(CustomerDataset, self).__init__(
                dataroot, annpath, trans_func, mode)
        self.lb_ignore = 255

        self.to_tensor = T.ToTensor(
            mean=(0.4, 0.4, 0.4), # rgb
            std=(0.2, 0.2, 0.2),
        )


class IndoorRiskDataset(BaseDataset):

    def __init__(self, dataroot, annpath, trans_func=None, mode='train'):
        super(IndoorRiskDataset, self).__init__(
                dataroot, annpath, trans_func, mode)
        self.lb_ignore = 255

        self.to_tensor = T.ToTensor(
            mean=(0.52594033, 0.46734318, 0.41189465), # indoor risk, rgb
            std=(0.24811083, 0.24959911, 0.25970083),
        )


