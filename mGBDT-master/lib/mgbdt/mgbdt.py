import numpy as np
import six

from lib.mgbdt.layer import BPLayer, TPLayer
from lib.mgbdt.loss import get_loss
from lib.mgbdt.model import MultiXGBModel

from lib.mgbdt import metrics
from lib.mgbdt.utils.log_utils import logger


class MGBDT:
    def __init__(self, log, datasetName=None, loss=None, target_lr=0.1, epsilon=0.3, verbose=False):
        """
        Attributes
        ----------
        layers: list
            layers[0] is just a stub for convinience
            layers[1] is the first layer
            layers[M] is the last layer (M is the number of layers)
        """
        if loss is not None and isinstance(loss, six.string_types):
            self.loss = get_loss(loss)
        else:
            self.loss = loss
        self.target_lr = target_lr
        self.epsilon = epsilon
        self.verbose = verbose
        self.layers = [None]

        self.logger = log
        self.datasetName = datasetName

    @property
    def is_last_layer_bp(self):
        return isinstance(self.layers[-1], BPLayer)

    @property
    def n_layers(self):
        return len(self.layers) - 1

    def add_layer(self, layer_type, *args, **kwargs):
        if layer_type == "tp_layer":
            layer_class = TPLayer
        elif layer_type == "bp_layer":
            layer_class = BPLayer
        else:
            raise ValueError()
        layer = layer_class(*args, **kwargs)
        self.layers.append(layer)

    def forward(self, X, n_layers=None):
        M = self.n_layers if n_layers is None else n_layers
        layers = self.layers
        out = X
        for i in range(1, M + 1):
            out = layers[i].forward(out)
        return out

    def get_hiddens(self, X, n_layers=None):
        """
        Return
        ------
        H: ndarray, shape = [M + 1, ]
            M represent the number of layers
            H[0] is the inputs
            H[1] is the outputs of the first layer
            H[M] is the outputs of the last layer (the outputs)
        """
        M = self.n_layers if n_layers is None else n_layers
        layers = self.layers
        H = [None for _ in range(M + 1)]
        H[0] = X
        for i in range(1, M + 1):
            H[i] = layers[i].forward(H[i - 1])
        return H

    def init(self, X, n_rounds=1, learning_rate=None, max_depth=None, batch=0):
        self.logger.info(self.datasetName + "[init][start]")
        layers = self.layers
        M = self.n_layers
        params = {}
        if learning_rate is not None:
            params["learning_rate"] = learning_rate
        if max_depth is not None:
            params["max_depth"] = max_depth
        for i in range(1, M + 1):
            self.logger.debug("{} [init] layer={}".format(self.datasetName, i))
            np.random.seed(32)
            rand_out = np.random.randn(X.shape[0], layers[i].output_size)
            if isinstance(layers[i].F, MultiXGBModel):
                layers[i].F.fit(X, rand_out, num_boost_round=n_rounds, params=params.copy())
            X = layers[i].forward(X)
        self.logger.info(self.datasetName + "[init][end]")

    def fit_inverse_mapping(self, X):
        # self.logger.info("[fit_inverse_mapping][start] X.shape={}".format(X.shape))
        M = self.n_layers
        H = self.get_hiddens(X)
        for i in range(2, M + 1):
            layer = self.layers[i]
            if hasattr(layer, "fit_inverse_mapping"):
                layer.fit_inverse_mapping(H[i - 1], epsilon=self.epsilon)
        # self.logger.info("[fit_inverse_mapping][end]")

    def fit_forward_mapping(self, X, y):
        # self.logger.info("[fit_forward_mapping][start] X.shape={}, y.shape={}".format(X.shape, y.shape))
        layers = self.layers
        M = self.n_layers
        # 2.1 Compute hidden units in prediction
        # self.logger.info("2.1 Compute hidden units")
        H = self.get_hiddens(X)  # 每层GBDT的输出.
        # 2.2 Compute the targets
        # self.logger.info("2.2 Compute the targets")
        Ht = [None for _ in range(M + 1)]
        if self.is_last_layer_bp:
            Ht[M] = y
        else:
            gradient = self.loss.backward(H[M], y)  # 计算最后一层的F与y的损失函数
            Ht[M] = H[M] - self.target_lr * gradient  # 将损失更新到预测值中
        for i in range(M, 1, -1):
            if isinstance(layers[i], BPLayer):
                assert i == M, "Only last layer can be BackPropogation Layer. i={}".format(i)
                Ht[i - 1] = layers[i].backward(H[i - 1], Ht[i], self.target_lr)
            else:
                Ht[i - 1] = layers[i].backward(H[i - 1], Ht[i])  # 将更新后的预测值输入到G中进行预测
        # 2.3 Training feedward mapping
        # self.logger.info("2.3 Training feedward mapping")
        for i in range(1, M + 1):
            if i == 1:
                H = X
            else:
                H = layers[i - 1].forward(H)
            fit_inputs = H
            fit_targets = Ht[i]
            # self.logger.info("fit layer={}".format(i))
            layers[i].fit_forward_mapping(fit_inputs, fit_targets)
            # layers[i].fit_forward_mapping(fit_inputs + np.random.randn(*fit_inputs.shape) * self.epsilon, fit_targets)
        # self.logger.info("[fit_forward_mapping][end]")
        return H, Ht

    def fit(self, X, y, n_epochs=1, eval_sets=None, batch=0, callback=None, n_eval_epochs=1, eval_metric=None):
        if eval_sets is None:
            eval_sets = ()
        self.log_metric("[epoch={}/{}][train]".format(0, n_epochs), X, y, eval_metric)
        for (x_test, y_test) in eval_sets:
            self.log_metric("[epoch={}/{}][test]".format(0, n_epochs), x_test, y_test, eval_metric)
        batch = min(len(X), batch)
        if batch == 0:
            batch = len(X)
        for _epoch in range(1, n_epochs + 1):
            if batch < len(X):
                perm = np.random.permutation(len(X))
            else:
                perm = list(range(len(X)))
            for si in range(0, len(perm) - batch + 1, batch):
                rand_idx = perm[si: si + batch]
                self.fit_inverse_mapping(X[rand_idx])  # 把输入数据正向预测一次，在反向用G把F的输出当输入，把F的输入当输出进行训练
            for si in range(0, len(perm) - batch + 1, batch):
                rand_idx = perm[si: si + batch]
                H, Ht = self.fit_forward_mapping(X[rand_idx], y[rand_idx])
            if n_eval_epochs > 0 and _epoch % n_eval_epochs == 0:
                self.log_metric("[epoch={}/{}][train]".format(_epoch, n_epochs), X, y, eval_metric)
                for (x_test, y_test) in eval_sets:
                    self.log_metric("[epoch={}/{}][test]".format(_epoch, n_epochs), x_test, y_test, eval_metric)
            if callback is not None:
                callback(self, _epoch, X, y, eval_sets)
            si += batch

    def mse(self, a, b):
        return np.mean((a - b) ** 2)

    def log_metric(self, prefix, X, y, eval_metric):
        pred = self.forward(X)
        loss = self.calc_loss(pred, y)
        if eval_metric == "accuracy":
            from sklearn.metrics import accuracy_score
            score = accuracy_score(y, pred.argmax(axis=1))
        elif eval_metric == "precision":
            score = metrics.get_precision(y, pred.argmax(axis=1))
        elif eval_metric == "recall":
            score = metrics.get_recall(y, pred.argmax(axis=1))
        elif eval_metric == "f1":
            score = metrics.get_f1score(y, pred.argmax(axis=1))
        elif eval_metric == "f2":
            score = metrics.get_f2score(y, pred.argmax(axis=1))
        elif eval_metric == "all":
            score = metrics.get_all_scores(y, pred.argmax(axis=1))
        else:
            score = None
        if score is None:
            self.logger.info("{} {} loss={:.6f}".format(prefix, self.datasetName, loss))
        elif isinstance(score, dict):
            self.logger.info("{} {} loss={:.6f}, precision={:.6f}, recall={:.6f}, f1={:.6f}, f2={:.6f}".format(prefix,
                                                                                                               self.datasetName,
                                                                                                               loss,
                                                                                                               score["precision"],
                                                                                                               score["recall"],
                                                                                                               score["f1"],
                                                                                                               score["f2"]))
        else:
            self.logger.info("{} {} loss={:.6f}, score={:.6f}".format(prefix, self.datasetName, loss, score))

    def calc_loss(self, pred, target):
        try:
            if self.is_last_layer_bp:
                loss = self.layers[-1].calc_loss(pred, target)
            else:
                loss = self.loss.forward(pred, target)
        except Exception:
            try:
                loss = self.mse(pred, target)
            except Exception:
                return np.nan
        return loss

    # def log(self, msg, color=None):
    #     if color is None:
    #         if self.verbose:
    #             logger.info(msg)
    #     else:
    #         logger.info(colored("{}".format(msg), color))

    def __repr__(self):
        res = "\n(MGBDT STRUCTURE BEGIN)\n"
        res += "loss={}, target_lr={:.3f}, epsilon={:.3f}\n".format(self.loss, self.target_lr, self.epsilon)
        for li, layer in enumerate(self.layers):
            if li == 0:
                continue
            res += "[layer={}] {}\n".format(li, layer)
        res += "(MGBDT STRUCTURE END)"
        return res
