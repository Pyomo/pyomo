BlockVector
===========

Methods specific to :py:class:`pyomo.contrib.pynumero.sparse.block_vector.BlockVector`:

  * :py:meth:`~pyomo.contrib.pynumero.sparse.block_vector.BlockVector.set_block`
  * :py:meth:`~pyomo.contrib.pynumero.sparse.block_vector.BlockVector.get_block`
  * :py:meth:`~pyomo.contrib.pynumero.sparse.block_vector.BlockVector.block_sizes`
  * :py:meth:`~pyomo.contrib.pynumero.sparse.block_vector.BlockVector.get_block_size`
  * :py:meth:`~pyomo.contrib.pynumero.sparse.block_vector.BlockVector.is_block_defined`
  * :py:meth:`~pyomo.contrib.pynumero.sparse.block_vector.BlockVector.copyfrom`
  * :py:meth:`~pyomo.contrib.pynumero.sparse.block_vector.BlockVector.copyto`
  * :py:meth:`~pyomo.contrib.pynumero.sparse.block_vector.BlockVector.copy_structure`
  * :py:meth:`~pyomo.contrib.pynumero.sparse.block_vector.BlockVector.set_blocks`
  * :py:meth:`~pyomo.contrib.pynumero.sparse.block_vector.BlockVector.pprint`

Attributes specific to :py:class:`pyomo.contrib.pynumero.sparse.block_vector.BlockVector`:

  * :py:attr:`~pyomo.contrib.pynumero.sparse.block_vector.BlockVector.nblocks`
  * :py:attr:`~pyomo.contrib.pynumero.sparse.block_vector.BlockVector.bshape`
  * :py:attr:`~pyomo.contrib.pynumero.sparse.block_vector.BlockVector.has_none`


NumPy compatible methods:

  * `numpy.ndarray.dot() <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.dot.html>`_
  * `numpy.ndarray.sum() <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.sum.html>`_
  * `numpy.ndarray.all() <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.all.html>`_
  * `numpy.ndarray.any() <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.any.html>`_
  * `numpy.ndarray.max() <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.max.html>`_
  * `numpy.ndarray.astype() <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.astype.html>`_
  * `numpy.ndarray.clip() <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.clip.html>`_
  * `numpy.ndarray.compress() <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.compress.html>`_
  * `numpy.ndarray.conj() <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.conj.html>`_
  * `numpy.ndarray.conjugate() <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.conjugate.html>`_
  * `numpy.ndarray.nonzero() <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.nonzero.html>`_
  * `numpy.ndarray.ptp() <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.ptp.html>`_
  * `numpy.ndarray.round() <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.round.html>`_
  * `numpy.ndarray.std() <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.std.html>`_
  * `numpy.ndarray.var() <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.var.html>`_
  * `numpy.ndarray.tofile() <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.tofile.html>`_
  * `numpy.ndarray.min() <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.min.html>`_
  * `numpy.ndarray.mean() <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.mean.html>`_
  * `numpy.ndarray.prod() <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.prod.html>`_
  * `numpy.ndarray.fill() <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.fill.html>`_
  * `numpy.ndarray.tolist() <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.tolist.html>`_
  * `numpy.ndarray.flatten() <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.flatten.html>`_
  * `numpy.ndarray.ravel() <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.ravel.html>`_
  * `numpy.ndarray.argmax() <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.argmax.html>`_
  * `numpy.ndarray.argmin() <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.argmin.html>`_
  * `numpy.ndarray.cumprod() <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.cumprod.html>`_
  * `numpy.ndarray.cumsum() <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.cumsum.html>`_
  * `numpy.ndarray.copy() <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.copy.html>`_

For example,

.. code-block:: python

  >>> import numpy as np
  >>> from pyomo.contrib.pynumero.sparse import BlockVector
  >>> v = BlockVector(2)
  >>> v.set_block(0, np.random.normal(size=100))
  >>> v.set_block(1, np.random.normal(size=30))
  >>> avg = v.mean()

NumPy compatible functions:

  * `numpy.log10() <https://numpy.org/doc/stable/reference/generated/numpy.log10.html>`_
  * `numpy.sin() <https://numpy.org/doc/stable/reference/generated/numpy.sin.html>`_
  * `numpy.cos() <https://numpy.org/doc/stable/reference/generated/numpy.cos.html>`_
  * `numpy.exp() <https://numpy.org/doc/stable/reference/generated/numpy.exp.html>`_
  * `numpy.ceil() <https://numpy.org/doc/stable/reference/generated/numpy.ceil.html>`_
  * `numpy.floor() <https://numpy.org/doc/stable/reference/generated/numpy.floor.html>`_
  * `numpy.tan() <https://numpy.org/doc/stable/reference/generated/numpy.tan.html>`_
  * `numpy.arctan() <https://numpy.org/doc/stable/reference/generated/numpy.arctan.html>`_
  * `numpy.arcsin() <https://numpy.org/doc/stable/reference/generated/numpy.arcsin.html>`_
  * `numpy.arccos() <https://numpy.org/doc/stable/reference/generated/numpy.arccos.html>`_
  * `numpy.sinh() <https://numpy.org/doc/stable/reference/generated/numpy.sinh.html>`_
  * `numpy.cosh() <https://numpy.org/doc/stable/reference/generated/numpy.cosh.html>`_
  * `numpy.abs() <https://numpy.org/doc/stable/reference/generated/numpy.absolute.html>`_
  * `numpy.tanh() <https://numpy.org/doc/stable/reference/generated/numpy.tanh.html>`_
  * `numpy.arccosh() <https://numpy.org/doc/stable/reference/generated/numpy.arccosh.html>`_
  * `numpy.arcsinh() <https://numpy.org/doc/stable/reference/generated/numpy.arcsinh.html>`_
  * `numpy.arctanh() <https://numpy.org/doc/stable/reference/generated/numpy.arctanh.html>`_
  * `numpy.fabs() <https://numpy.org/doc/stable/reference/generated/numpy.fabs.html>`_
  * `numpy.sqrt() <https://numpy.org/doc/stable/reference/generated/numpy.sqrt.html>`_
  * `numpy.log() <https://numpy.org/doc/stable/reference/generated/numpy.log.html>`_
  * `numpy.log2() <https://numpy.org/doc/stable/reference/generated/numpy.log2.html>`_
  * `numpy.absolute() <https://numpy.org/doc/stable/reference/generated/numpy.absolute.html>`_
  * `numpy.isfinite() <https://numpy.org/doc/stable/reference/generated/numpy.isfinite.html>`_
  * `numpy.isinf() <https://numpy.org/doc/stable/reference/generated/numpy.isinf.html>`_
  * `numpy.isnan() <https://numpy.org/doc/stable/reference/generated/numpy.isnan.html>`_
  * `numpy.log1p() <https://numpy.org/doc/stable/reference/generated/numpy.log1p.html>`_
  * `numpy.logical_not() <https://numpy.org/doc/stable/reference/generated/numpy.logical_not.html>`_
  * `numpy.expm1() <https://numpy.org/doc/stable/reference/generated/numpy.expm1.html>`_
  * `numpy.exp2() <https://numpy.org/doc/stable/reference/generated/numpy.exp2.html>`_
  * `numpy.sign() <https://numpy.org/doc/stable/reference/generated/numpy.sign.html>`_
  * `numpy.rint() <https://numpy.org/doc/stable/reference/generated/numpy.rint.html>`_
  * `numpy.square() <https://numpy.org/doc/stable/reference/generated/numpy.square.html>`_
  * `numpy.positive() <https://numpy.org/doc/stable/reference/generated/numpy.positive.html>`_
  * `numpy.negative() <https://numpy.org/doc/stable/reference/generated/numpy.negative.html>`_
  * `numpy.rad2deg() <https://numpy.org/doc/stable/reference/generated/numpy.rad2deg.html>`_
  * `numpy.deg2rad() <https://numpy.org/doc/stable/reference/generated/numpy.deg2rad.html>`_
  * `numpy.conjugate() <https://numpy.org/doc/stable/reference/generated/numpy.conjugate.html>`_
  * `numpy.reciprocal() <https://numpy.org/doc/stable/reference/generated/numpy.reciprocal.html>`_
  * `numpy.signbit() <https://numpy.org/doc/stable/reference/generated/numpy.signbit.html>`_
  * `numpy.add() <https://numpy.org/doc/stable/reference/generated/numpy.add.html>`_
  * `numpy.multiply() <https://numpy.org/doc/stable/reference/generated/numpy.multiply.html>`_
  * `numpy.divide() <https://numpy.org/doc/stable/reference/generated/numpy.divide.html>`_
  * `numpy.subtract() <https://numpy.org/doc/stable/reference/generated/numpy.subtract.html>`_
  * `numpy.greater() <https://numpy.org/doc/stable/reference/generated/numpy.greater.html>`_
  * `numpy.greater_equal() <https://numpy.org/doc/stable/reference/generated/numpy.greater_equal.html>`_
  * `numpy.less() <https://numpy.org/doc/stable/reference/generated/numpy.less.html>`_
  * `numpy.less_equal() <https://numpy.org/doc/stable/reference/generated/numpy.less_equal.html>`_
  * `numpy.not_equal() <https://numpy.org/doc/stable/reference/generated/numpy.not_equal.html>`_
  * `numpy.maximum() <https://numpy.org/doc/stable/reference/generated/numpy.maximum.html>`_
  * `numpy.minimum() <https://numpy.org/doc/stable/reference/generated/numpy.minimum.html>`_
  * `numpy.fmax() <https://numpy.org/doc/stable/reference/generated/numpy.fmax.html>`_
  * `numpy.fmin() <https://numpy.org/doc/stable/reference/generated/numpy.fmin.html>`_
  * `numpy.equal() <https://numpy.org/doc/stable/reference/generated/numpy.equal.html>`_
  * `numpy.logical_and() <https://numpy.org/doc/stable/reference/generated/numpy.logical_and.html>`_
  * `numpy.logical_or() <https://numpy.org/doc/stable/reference/generated/numpy.logical_or.html>`_
  * `numpy.logical_xor() <https://numpy.org/doc/stable/reference/generated/numpy.logical_xor.html>`_
  * `numpy.logaddexp() <https://numpy.org/doc/stable/reference/generated/numpy.logaddexp.html>`_
  * `numpy.logaddexp2() <https://numpy.org/doc/stable/reference/generated/numpy.logaddexp2.html>`_
  * `numpy.remainder() <https://numpy.org/doc/stable/reference/generated/numpy.remainder.html>`_
  * `numpy.heaviside() <https://numpy.org/doc/stable/reference/generated/numpy.heaviside.html>`_
  * `numpy.hypot() <https://numpy.org/doc/stable/reference/generated/numpy.hypot.html>`_

For example,

.. code-block:: python

  >>> import numpy as np
  >>> from pyomo.contrib.pynumero.sparse import BlockVector
  >>> v = BlockVector(2)
  >>> v.set_block(0, np.random.normal(size=100))
  >>> v.set_block(1, np.random.normal(size=30))
  >>> inf_norm = np.max(np.abs(v))

.. autoclass:: pyomo.contrib.pynumero.sparse.block_vector.BlockVector
.. automethod:: pyomo.contrib.pynumero.sparse.block_vector.BlockVector.set_block
.. automethod:: pyomo.contrib.pynumero.sparse.block_vector.BlockVector.get_block
.. automethod:: pyomo.contrib.pynumero.sparse.block_vector.BlockVector.block_sizes
.. automethod:: pyomo.contrib.pynumero.sparse.block_vector.BlockVector.get_block_size
.. automethod:: pyomo.contrib.pynumero.sparse.block_vector.BlockVector.is_block_defined
.. automethod:: pyomo.contrib.pynumero.sparse.block_vector.BlockVector.copyfrom
.. automethod:: pyomo.contrib.pynumero.sparse.block_vector.BlockVector.copyto
.. automethod:: pyomo.contrib.pynumero.sparse.block_vector.BlockVector.copy_structure
.. automethod:: pyomo.contrib.pynumero.sparse.block_vector.BlockVector.set_blocks
.. automethod:: pyomo.contrib.pynumero.sparse.block_vector.BlockVector.pprint
.. autoproperty:: pyomo.contrib.pynumero.sparse.block_vector.BlockVector.nblocks
.. autoproperty:: pyomo.contrib.pynumero.sparse.block_vector.BlockVector.bshape
.. autoproperty:: pyomo.contrib.pynumero.sparse.block_vector.BlockVector.has_none
