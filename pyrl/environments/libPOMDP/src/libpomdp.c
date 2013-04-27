
#include "Python.h" 
#include "arrayobject.h"
#include "noprefix.h"

//#include "numpy/arrayobject.h"
//#include "numpy/noprefix.h"

#include "libpomdp.h"


static PyMethodDef libPOMDPMethods[] = {
  {"readMDP", C_readMDP, METH_VARARGS,"Load an MDP/POMDP specification file."},
  {"getSparseTransitionMatrix", getSparseTransitionMatrix, METH_VARARGS,"Return the transition matrix as a list (for each action) of row,col,data formatted sparse matrix."},
  {"getSparseObsMatrix", getSparseObsMatrix, METH_VARARGS,"Return the observation matrix as a list (for each action) of row,col,data formatted sparse matrix."},
  {"getReward", C_getReward, METH_VARARGS,"Get immediate reward"},
  {"transformBelief", C_transformBelief, METH_VARARGS,"Update the belief state"},
  {"getInitialBelief", C_getInitialBelief, METH_VARARGS,"Get initial belief state of the system."},
  {"isRewardType", C_isRewardType, METH_VARARGS,"Returns 1 if the specification is for reward, 0 if for cost."},
  {"getNumObservations", C_getNumObservations, METH_VARARGS,"Get the number of observations"},
  {"getNumActions", C_getNumActions, METH_VARARGS,"Get the number of actions."},
  {"getNumStates", C_getNumStates, METH_VARARGS,"Get the number of states"},
  {"getDiscount", C_getDiscount, METH_VARARGS,"Get the discount factor."},
  {"getRewardRange", C_getRewardRange, METH_VARARGS,"Get reward range: min, max."},
  {"loadfile", loadFile, METH_VARARGS,"Load POMDP/MDP from a file..."},
  {NULL, NULL, 0, NULL}
};

void initlibpomdp() { 
  (void) Py_InitModule("libpomdp", libPOMDPMethods);
  import_array(); 
}


static PyObject *C_readMDP(PyObject *self, PyObject *args) {
  const char *filename; 
  int res;

  if (!PyArg_ParseTuple(args, "s", &filename)) {
    return NULL;
  }

  printf("Input filename string: %s\n", filename);
  res = readMDP(filename);
  return Py_BuildValue("i", res);
}

static PyObject *C_getRewardRange(PyObject *self, PyObject *args) {
  if (P == NULL) {
    return NULL;
  }

  return Py_BuildValue("dd", gMinReward, gMaxReward);
}

static PyObject *C_getDiscount(PyObject *self, PyObject *args) {

  if (P == NULL) {
    return NULL;
  }

  return Py_BuildValue("d", gDiscount);
}

static PyObject *C_getNumStates(PyObject *self, PyObject *args) {

  if (P == NULL) {
    return NULL;
  }

  return Py_BuildValue("i", gNumStates);
}

static PyObject *C_getNumActions(PyObject *self, PyObject *args) {

  if (P == NULL) {
    return NULL;
  }

  return Py_BuildValue("i", gNumActions);
}


static PyObject *C_getNumObservations(PyObject *self, PyObject *args) {

  if (P == NULL) {
    return NULL;
  }

  return Py_BuildValue("i", gNumObservations);
}

static PyObject *C_isRewardType(PyObject *self, PyObject *args) {

  int isReward = 0;
  if (P == NULL) {
    return NULL;
  }
  if (gValueType == REWARD_value_type)
    isReward = 1;

  return Py_BuildValue("i", isReward);
}



static PyObject *C_getInitialBelief(PyObject *self, PyObject *args) {

  PyArrayObject *vout; 
  npy_intp dims[1];
  int i;

  if (P == NULL) {
    return NULL;
  }

  dims[0] = gNumStates;

  vout = (PyArrayObject *) PyArray_SimpleNew(1, dims, NPY_DOUBLE); 
  for (i = 0; i < gNumStates; i++) {
    *(double*)(vout->data + i*(vout->strides[0])) = gInitialBelief[i];
  }

  return Py_BuildValue("O", vout);
}

static PyObject *C_transformBelief(PyObject *self, PyObject *args) {

  PyArrayObject *vout,*vin; 
  double *pi, *pi_hat;
  int a, obs;
  npy_intp dims[1];
  int i;

  if (P == NULL || !PyArg_ParseTuple(args, "Oii", &vin, &a, &obs))  return NULL;

  dims[0] = gNumStates;
  pi = (double*)malloc(gNumStates*sizeof(double));
  pi_hat = (double*)malloc(gNumStates*sizeof(double));

  for (i = 0; i < gNumStates; i++) {
    pi[i] = *(double*)(vin->data + i*vin->strides[0]);
  }

  if (!transformBeliefState(pi, pi_hat, a, obs)) return NULL;

  vout = (PyArrayObject *) PyArray_SimpleNew(1, dims, NPY_DOUBLE); 
  for (i = 0; i < gNumStates; i++) {
    *(double*)(vout->data + i*(vout->strides[0])) = pi_hat[i];
  }

  return Py_BuildValue("O", vout);
}

static PyObject *C_getReward(PyObject *self, PyObject *args) {
  int state, next_state, a, obs;
  double reward;

  if (P == NULL || 
      !PyArg_ParseTuple(args, "iiii", &state, &a, &next_state, &obs) ||
      !validState(state) || !validState(next_state) || !validAction(a) || 
      !validObservation(obs))  {
    PyErr_SetString(PyExc_ValueError,"Value passed was out of valid range.");
    return NULL;
  }

  reward = getImmediateReward(a, state, next_state, obs);
  return Py_BuildValue("d", reward);
}

static PyObject* getSparseTransitionMatrix(PyObject *self, PyObject *args) {
  PyObject *Pobj;

  if (P == NULL) {
    return NULL;
  }

  Pobj = fillPyMatrix(P);
  
  return Py_BuildValue("O", Pobj);
}

static PyObject* getSparseObsMatrix(PyObject *self, PyObject *args) {
  PyObject *Pobj;

  if (P == NULL) {
    return NULL;
  }

  Pobj = fillPyMatrix(R);
  return Py_BuildValue("O", Pobj);
}



static PyObject *loadFile(PyObject *self, PyObject *args) {
  /*  PyArrayObject *matin, *matout;
      double **cin, **cout; */
  const char *filename; 
  PyObject *Pobj;
  PyObject *Robj;
  int isReward;

  if (!PyArg_ParseTuple(args, "s", &filename)) {
    return NULL;
  }

  printf("Input filename string: %s\n", filename);
  readMDP(filename);

  if (gValueType == REWARD_value_type)
    isReward = 1;
  else
    isReward = 0;

  Pobj = fillPyMatrix(P);
  Robj = fillPyMatrix(R);
  
  return Py_BuildValue("diOO", gDiscount, isReward, Pobj, Robj);
}



PyObject* fillPyMatrix(Matrix *target) {
  PyArrayObject *matout; 
  npy_intp dims[2];
  int a, i, j;
  PyObject *obj = PyList_New(gNumActions);
  dims[0] = 3;

  for(a = 0; a < gNumActions; a++) {
    dims[1] = target[a]->num_non_zero;
    matout = (PyArrayObject *) PyArray_SimpleNew(2, dims, NPY_DOUBLE); 
    /* rows, cols, data */
    for (i = 0; i < target[a]->num_rows; i++) {
      for (j = 0; j < target[a]->row_length[i]; j++) {
	*(double*)(matout->data + 0*(matout->strides[0]) + 
		   (target[a]->row_start[i] + j)*matout->strides[1]) = i;
      }
    }

    for (i = 0; i < dims[1]; i++) {
	*(double*)(matout->data + 1*(matout->strides[0]) + i*matout->strides[1]) = target[a]->col[i];
	*(double*)(matout->data + 2*(matout->strides[0]) + i*matout->strides[1]) = target[a]->mat_val[i];
    }

    PyList_SetItem(obj, a, matout);    
  }

  return obj;
}

