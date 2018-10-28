"""
Tuning MNLI
"""

from libs.models.NLIRNN import NLIRNN
from config.constants import HyperParamKey, ControlKey, OutputKey
import logging

logger = logging.getLogger('__main__')


class MNLIRNN(NLIRNN):
    def __init__(self, hparams, lparams, cparams, label='scratch', nolog=False):
        super(MNLIRNN, self).__init__(hparams, lparams, cparams, label, nolog)
        self.active_genre = None

    def activate_genre(self, genre):
        self.active_genre = genre

    def train(self, loader, tqdm_handler):
        """
        overloading the train method on the BaseLoader due to difference in parameters
        :param loader: torch.utils.DataLoader, filled with data in the load routine
        :param tqdm_handler:
        :return:
        """
        if self.model is None:
            logger.error("Cannot train an uninitialized model - stopping training on model %s" % self.label)
        else:
            # ------ basic training loop ------
            self._init_optim_and_scheduler()
            criterion = self.hparams[HyperParamKey.CRITERION]()

            early_stop_training = False
            ni_buffer = 0
            # for epoch in tqdm_handler(range(self.hparams[HyperParamKey.NUM_EPOCH] - self.cur_epoch)):
            for epoch in range(self.hparams[HyperParamKey.NUM_EPOCH] - self.cur_epoch):
                self.scheduler.step(epoch=self.cur_epoch)  # scheduler calculates the lr based on the cur_epoch
                self.cur_epoch += 1
                logger.info("stepped scheduler to epoch = %s" % str(self.scheduler.last_epoch + 1))

                for i, (sent1_batch,
                        sent2_batch,
                        len1_batch,
                        len2_batch,
                        label_batch) in enumerate(loader.loaders['train'][self.active_genre]):
                    self.model.train()  # good practice to set the model to training mode (enables dropout)
                    self.optim.zero_grad()
                    outputs = self.model(sent1_batch,
                                         sent2_batch,
                                         len1_batch,
                                         len2_batch)  # forward pass
                    loss = criterion(outputs, label_batch)          # computing loss
                    loss.backward()                                 # backprop
                    self.optim.step()                               # taking a step

                    # --- Model Evaluation Iteration ---
                    is_best = False
                    if (i + 1) % self.hparams[HyperParamKey.TRAIN_LOOP_EVAL_FREQ] == 0:
                        val_acc, val_loss = self.eval_model(loader.loaders['val'][self.active_genre])
                        train_acc, train_loss = self.eval_model(loader.loaders['train'][self.active_genre])
                        iter_curve = self.iter_curves[self.VAL_ACC]
                        if len(iter_curve) > 0 and val_acc >= max(iter_curve):
                            is_best = True

                        logger.info('Ep:%s/%s, Bt:%s/%s, VAcc:%.2f, VLoss:%.1f, TAcc:%.2f, TLoss:%.1f, LR:%.4f' %
                                    (self.cur_epoch,
                                     self.hparams[HyperParamKey.NUM_EPOCH],
                                     i + 1,
                                     len(loader.loaders['train'][self.active_genre]),
                                     val_acc,
                                     val_loss,
                                     train_acc,
                                     train_loss,
                                     self.optim.param_groups[0]['lr'])  # assumes a constant lr across params
                                    )
                        self.iter_curves[self.TRAIN_LOSS].append(train_loss)
                        self.iter_curves[self.TRAIN_ACC].append(train_acc)
                        self.iter_curves[self.VAL_LOSS].append(val_loss)
                        self.iter_curves[self.VAL_ACC].append(val_acc)
                        if self.cparams[ControlKey.SAVE_BEST_MODEL] and is_best:
                            self.save(fn=self.BEST_FN)

                        # reporting back up to output_dict
                        if is_best:
                            self.output_dict[OutputKey.BEST_VAL_ACC] = val_acc
                            self.output_dict[OutputKey.BEST_VAL_LOSS] = val_loss

                        # decrease the LR if we haven't see improvement in a while
                        no_improvement = self.check_no_improvement()
                        if no_improvement and self.hparams[HyperParamKey.DECAY_LR_NO_IMPROV] < 1.0 and ni_buffer <= 0:
                            # setting lr on the schedulers
                            for j, base_lr in enumerate(self.scheduler.base_lrs):
                                self.scheduler.base_lrs[j] = base_lr * self.hparams[HyperParamKey.DECAY_LR_NO_IMPROV]

                            # setting lr on the optimizers
                            for param_group, lr in zip(self.scheduler.optimizer.param_groups, self.scheduler.get_lr()):
                                param_group['lr'] = lr

                            logger.info('reducing base_lr to %.5f since no improvement observed in %s steps' % (
                                self.scheduler.base_lrs[0],
                                self.hparams[HyperParamKey.NO_IMPROV_LOOK_BACK]
                            ))

                            ni_buffer = self.hparams[HyperParamKey.NO_IMPROV_LOOK_BACK]
                        ni_buffer -= 1

                        if self.hparams[HyperParamKey.CHECK_EARLY_STOP]:
                            early_stop_training = self.check_early_stop()
                        if early_stop_training:
                            logger.info('--- stopping training due to early stop ---')
                            break
                    if early_stop_training:
                        break

                # appending to epock trackers
                val_acc, val_loss = self.eval_model(loader.loaders['val'][self.active_genre])
                train_acc, train_loss = self.eval_model(loader.loaders['train'][self.active_genre])
                self.epoch_curves[self.TRAIN_LOSS].append(train_loss)
                self.epoch_curves[self.TRAIN_ACC].append(train_acc)
                self.epoch_curves[self.VAL_LOSS].append(val_loss)
                self.epoch_curves[self.VAL_ACC].append(val_acc)
                if self.cparams[ControlKey.SAVE_EACH_EPOCH]:
                    self.save()
                if early_stop_training:
                    break

            # final loss reporting
            val_acc, val_loss = self.eval_model(loader.loaders['val'][self.active_genre])
            train_acc, train_loss = self.eval_model(loader.loaders['train'][self.active_genre])
            self.output_dict[OutputKey.FINAL_TRAIN_ACC] = train_acc
            self.output_dict[OutputKey.FINAL_TRAIN_LOSS] = train_loss
            self.output_dict[OutputKey.FINAL_VAL_ACC] = val_acc
            self.output_dict[OutputKey.FINAL_VAL_LOSS] = val_loss
            logger.info("training completed, results collected ...")
