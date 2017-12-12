import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions

from chainer.datasets import get_cifar10

from vlad_pooling import VLADpooling
from attention_vlad_pooling import AttentionVLADpooling


class Net(chainer.Chain):
    def __init__(self, n_class):
        super(Net, self).__init__()
        with self.init_scope():
            self.vlad  = VLADpooling(3, 64)
            self.avlad = AttentionVLADpooling(3, 64)
            self.fc    = L.Linear(64 * 3 * 2, n_class)

    def __call__(self, x):
        return self.fc(F.concat((self.vlad(x), self.avlad(x)), axis=1))


def main():
    class_label = 10
    train, test = get_cifar10()

    model = L.Classifier(Net(class_label))

    optimizer = chainer.optimizers.MomentumSGD(0.05)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(5e-4))

    train_iter = chainer.iterators.SerialIterator(train, 64)
    test_iter  = chainer.iterators.SerialIterator(test , 64,
        repeat=False, shuffle=False)

    stop_trigger = (30, 'epoch')

    updater = training.StandardUpdater(train_iter, optimizer, device=-1)
    trainer = training.Trainer(updater, stop_trigger, out='out')

    trainer.extend(extensions.Evaluator(test_iter, model, device=-1))

    trainer.extend(
        extensions.ExponentialShift('lr', 0.5), trigger=(25, 'epoch'))

    trainer.extend(extensions.snapshot(), trigger=(30, 'epoch'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
    trainer.extend(extensions.ProgressBar(
        training_length=(30, 'epoch')))

    trainer.run()


if __name__ == '__main__':
    main()
