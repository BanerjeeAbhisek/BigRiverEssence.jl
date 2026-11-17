"""Generate Gaussian MOFA reference outputs with the original mofapy2 nodes."""

from pathlib import Path
import contextlib
import copy
import csv
import io
import platform
import numpy as np
import scipy
from mofapy2.build_model.init_model import initModel
from mofapy2.build_model.build_model import buildBiofam
from mofapy2.build_model.utils import process_data
from mofapy2.core.BayesNet import BayesNet


folder = Path(__file__).resolve().parent
np.random.seed(160)
n, k = 12, 2
groups = np.tile([0, 1], n // 2)
latent = np.random.normal(size=(n, k))
views = [latent @ np.random.normal(size=(k, p)) +
         0.1 * np.random.normal(size=(n, p)) for p in (5, 3)]
for view in views:
    view[2, :] = np.nan  # one sample missing across both views
views[0][groups == 1, 4] = np.nan  # an entirely missing group/feature
views[1][6, 1] = np.nan
z_init = np.random.normal(size=(n, k))
for m, view in enumerate(views, 1):
    np.savetxt(folder / f"X{m}.csv", view, delimiter=",")
np.savetxt(folder / "Z_init.csv", z_init, delimiter=",")
np.savetxt(folder / "groups.csv", groups, delimiter=",", fmt="%d")

metadata = []
for aw in (False, True):
    for az in (False, True):
        for sw in (False, True):
            case = f"aw{int(aw)}_az{int(az)}_sw{int(sw)}"
            target = folder / case
            target.mkdir(exist_ok=True)
            data_opts = dict(center_groups=True, scale_views=False,
                             scale_groups=False, groups_names=[0, 1], samples_groups=groups)
            data = process_data([x.copy() for x in views], ["gaussian"] * 2,
                                data_opts, groups)
            dim = dict(N=n, K=k, M=2, D=[5, 3], G=2)
            model_opts = dict(ard_weights=aw, ard_factors=az,
                              spikeslab_weights=sw, spikeslab_factors=False,
                              likelihoods=["gaussian"] * 2)
            init = initModel(dim, data, ["gaussian"] * 2, groups, seed=160)
            init.initZ(qmean=z_init.copy())
            if sw:
                init.initSW(qmean_S1=0)
                init.initThetaW()
            else:
                init.initW(qmean=0)
            if aw:
                init.initAlphaW()
            if az:
                init.initAlphaZ()
            init.initTau()
            init.initY()
            builder = buildBiofam(data, dim, data_opts, model_opts, {})
            builder.init_model = init
            builder.createMarkovBlankets()
            nodes = init.getNodes()
            schedule = ["Y", "W"] + (["AlphaW"] if aw else []) + \
                (["ThetaW"] if sw else []) + ["Z"] + (["AlphaZ"] if az else []) + ["Tau"]
            options = dict(maxiter=31, start_drop=1, freq_drop=1,
                           drop=dict(min_r2=None), verbose=False, quiet=True,
                           convergence_mode="fast", forceiter=True,
                           schedule=schedule, start_sparsity=3, gpu_mode=False,
                           start_elbo=1, freqELBO=1, min_iter=0,
                           stochastic=False, weight_views=False)
            net = BayesNet(dim, nodes)
            net.setTrainOptions(options)
            stopping_net = copy.deepcopy(net)
            with contextlib.redirect_stdout(io.StringIO()):
                net.iterate()  # original training loop; 30 actual update sweeps
            terms = net.getTrainingStats()["elbo_terms"]
            columns = ["Y", "W", "Z", "Tau", "AlphaW", "AlphaZ", "ThetaW"]
            values = np.column_stack([terms[c].values[:31] if c in terms else np.zeros(31)
                                      for c in columns])
            # The reference allocates spare unused rows in its history.
            np.savetxt(target / "elbo.csv", np.column_stack([
                np.arange(31), values, net.getTrainingStats()["elbo"][:31]]), delimiter=",")
            arrays = [nodes["Z"].getExpectation(), nodes["Z"].Q.getParameters()["var"]]
            for m in range(2):
                arrays.extend([nodes["W"].getNodes()[m].getExpectation(),
                               nodes["W"].getNodes()[m].getExpectations()["E2"],
                               nodes["Tau"].getNodes()[m].Q.getParameters()["a"],
                               nodes["Tau"].getNodes()[m].Q.getParameters()["b"]])
            if aw:
                for node in nodes["AlphaW"].getNodes():
                    arrays.extend([node.Q.getParameters()["a"], node.Q.getParameters()["b"]])
            if az:
                arrays.extend([nodes["AlphaZ"].Q.getParameters()["a"],
                               nodes["AlphaZ"].Q.getParameters()["b"]])
            if sw:
                for m in range(2):
                    theta = nodes["ThetaW"].getNodes()[m].Q.getParameters()
                    arrays.extend([nodes["W"].getNodes()[m].getExpectations()["EB"],
                                   theta["a"], theta["b"]])
            np.savetxt(target / "posterior.csv", np.concatenate([
                np.asarray(a).ravel(order="F") for a in arrays]), delimiter=",")
            np.savetxt(target / "r2.csv", net.calculate_variance_explained(total=True), delimiter=",")
            np.savetxt(target / "factor_r2.csv", np.asarray(
                net.calculate_variance_explained()).ravel(order="F"), delimiter=",")
            # Also compare the unmodified reference's early-stopping behavior.
            stopping_options = dict(options, maxiter=1001, forceiter=False, freqELBO=5)
            stopping_net.setTrainOptions(stopping_options)
            with contextlib.redirect_stdout(io.StringIO()):
                stopping_net.iterate()
            history = stopping_net.getTrainingStats()["elbo"]
            converged = len(history) < stopping_options["maxiter"] + 1
            # On convergence the reference slices history before the stopping
            # row, although its parameter state includes that update sweep.
            sweeps = len(history) if converged else 1000
            np.savetxt(target / "stopping.csv", [sweeps, int(converged),
                       stopping_net.calculateELBO()["total"]], delimiter=",")
            metadata.append([case, int(aw), int(az), int(sw), 30, 3])

with (folder / "meta.csv").open("w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(["case", "ard_weights", "ard_factors", "spikeslab_weights", "sweeps", "start_sparsity"])
    writer.writerows(metadata)
with (folder / "session_info.txt").open("w") as stream:
    stream.write(f"Python: {platform.python_version()}\nNumPy: {np.__version__}\nSciPy: {scipy.__version__}\n")
    stream.write("mofapy2: 0.7.5\nSource: 90418e5021b3ae735ebf1fea5d7e07cae71c8bb7\n")
