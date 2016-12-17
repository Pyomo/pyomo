from MatrixForm import to_matrix_form
from runpiecewisend import model
import numpy
import scipy

model.nonlinear.deactivate()

(c0, c,
 bL, bU,
 (A_data, A_indices, A_indptr),
 xL, xU,
 vartocol, contorow) = to_matrix_form(model)

# convert to numpy arrays
c = numpy.array(c)
bL = numpy.array(bL)
bU = numpy.array(bU)
A_data = numpy.array(A_data)
A_indices = numpy.array(A_indices)
A_indptr = numpy.array(A_indptr)
xL = numpy.array(xL)
xU = numpy.array(xU)

# generate a scipy sparse matrix object
A = scipy.sparse.csr_matrix((A_data, A_indices, A_indptr))
print(A)
