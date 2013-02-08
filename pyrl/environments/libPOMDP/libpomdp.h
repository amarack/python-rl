

#include "Python.h" // Must be the first header?
#include "arrayobject.h" // for numpy arrays
#include "imm-reward.h"

#include <stdlib.h>
#include "mdp.h"

static PyObject *C_readMDP(PyObject *self, PyObject *args);
static PyObject* getSparseTransitionMatrix(PyObject *self, PyObject *args);
static PyObject* getSparseObsMatrix(PyObject *self, PyObject *args);


static PyObject *C_getReward(PyObject *self, PyObject *args);
static PyObject *C_transformBelief(PyObject *self, PyObject *args);
static PyObject *C_getInitialBelief(PyObject *self, PyObject *args);
static PyObject *C_isRewardType(PyObject *self, PyObject *args);
static PyObject *C_getNumObservations(PyObject *self, PyObject *args);
static PyObject *C_getNumActions(PyObject *self, PyObject *args);
static PyObject *C_getNumStates(PyObject *self, PyObject *args);
static PyObject *C_getDiscount(PyObject *self, PyObject *args);

static PyObject *loadFile(PyObject *self, PyObject *args); 
PyObject* fillPyMatrix(Matrix *target);

#define validState(S) (S >= 0 && S < gNumStates)
#define validAction(A) (A >= 0 && A < gNumActions)
#define validObservation(O) (O >= 0 && O < gNumObservations)

