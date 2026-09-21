"""Regression test for the caller-reference memory bug fixed in
models/static_lgbm.py::train_from_holder(). See that function's docstring
for the full mechanism: measured directly (weakref + RssAnon, Python
3.10.12), a caller's own reference to a large array survives for the
ENTIRE duration of any call where it's a positional argument -- true
whether the caller binds it to a name first or unpacks a producer's
return value inline (f(*g())), since CPython's caller-side evaluation
stack retains the reference regardless of calling convention.
train_from_holder() avoids this by popping arrays out of a holder dict
the caller never separately names, so its own `del` genuinely drops the
last reference.

This proves the fix actually works, not just that it looks right: hold a
weakref to X_train/X_val taken before the call, delete this test's own
names (mirroring scripts/train_static.py::main(), which never binds
X_train/X_val to names of its own), and confirm both weakrefs are dead
after train_from_holder() returns -- i.e. the arrays were actually
collected, not merely out of scope by name. Tiny arrays / few rounds so
this stays fast.
"""
import gc
import weakref

import numpy as np

from models.static_lgbm import train_from_holder


def test_train_from_holder_frees_raw_arrays():
    n_train, n_val, n_feat = 2000, 500, 20
    rng = np.random.default_rng(0)
    X_train = rng.random((n_train, n_feat), dtype=np.float32)
    y_train = (rng.random(n_train) > 0.5).astype(np.int32)
    X_val = rng.random((n_val, n_feat), dtype=np.float32)
    y_val = (rng.random(n_val) > 0.5).astype(np.int32)

    w_train = weakref.ref(X_train)
    w_val = weakref.ref(X_val)

    holder = {"X_train": X_train, "y_train": y_train, "X_val": X_val, "y_val": y_val}
    # This test's own names must go too -- otherwise the test itself would
    # be the lingering reference the fix is designed to eliminate.
    del X_train, y_train, X_val, y_val

    model = train_from_holder(holder, n_estimators=2, early_stopping_rounds=1)

    gc.collect()
    assert w_train() is None, "X_train was not freed -- caller-reference bug has regressed"
    assert w_val() is None, "X_val was not freed -- caller-reference bug has regressed"
    assert holder == {}, "holder should be fully emptied by train_from_holder()"
    assert model.num_iterations >= 1
    assert model.calibrator is None  # calibrate() is a separate step, not part of this function
