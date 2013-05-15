
#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include "game.h"
#include "random.h"
#include "macros.h"
#include "file_tools.h"
#include "feature_functions.h"


Game *tetris_game = NULL;
FeaturePolicy *original_features = NULL;
FeaturePolicy *dellacherie_features = NULL;

/*
static PyObject * mdptetris_OriginalFeaturesRanges(PyObject *self, PyObject *args)
{
    PyObject * features_list = NULL;
    int num_features;
    int i;
    double **ranges = NULL;
    ranges = get_policy_feature_ranges(tetris_game, original_features);
    num_features = original_features->nb_features;

    features_list = PyList_New(num_features);
    for (i = 0 ; i < num_features; i++)
        PyList_SetItem(features_list,i,Py_BuildValue("(dd)", ranges[i][0], ranges[i][1]));
    return features_list;
}*/


static PyObject * mdptetris_ResetGame(PyObject *self, PyObject *args)
{
    game_reset(tetris_game);
    return Py_BuildValue("");
}

static PyObject * mdptetris_FeaturesRanges(PyObject *self, PyObject *args)
{
    PyObject * features_list = NULL;
    int num_features;
    char *featureset = NULL;
    int i;
    double **ranges = NULL;

    if (!PyArg_ParseTuple(args, "s", &featureset))
        return NULL;

    if (!strcmp(featureset, "original")) {
        ranges = get_policy_feature_ranges(tetris_game, original_features);
        num_features = original_features->nb_features;
    } else if (!strcmp(featureset, "dellacherie")) {
        ranges = get_policy_feature_ranges(tetris_game, dellacherie_features);
        num_features = dellacherie_features->nb_features;
    } else
        return Py_BuildValue("");

    features_list = PyList_New(num_features);

    for (i = 0 ; i < num_features; i++) {
         PyList_SetItem(features_list,i,Py_BuildValue("(dd)", ranges[i][0], ranges[i][1]));
         free(ranges[i]);
    }
    FREE(ranges);
    fflush(stdout);
    return features_list;
}

static PyObject * mdptetris_GetOriginalFeatures(PyObject *self, PyObject *args)
{
    PyObject * features_list = NULL;
    int num_features;
    int i;
    double *features = NULL;
    features = get_feature_values(tetris_game, original_features);
    num_features = original_features->nb_features;

    features_list = PyList_New(num_features);
    for (i = 0 ; i < num_features; i++)
        PyList_SetItem(features_list,i,PyFloat_FromDouble((double)features[i]));
    return features_list;
}

static PyObject * mdptetris_GetDellacherieFeatures(PyObject *self, PyObject *args)
{
    PyObject * features_list = NULL;
    int num_features;
    int i;
    double *features = NULL;
    features = get_feature_values(tetris_game, dellacherie_features);
    num_features = dellacherie_features->nb_features;

    features_list = PyList_New(num_features);
    for (i = 0 ; i < num_features; i++)
        PyList_SetItem(features_list,i,PyFloat_FromDouble((double)features[i]));
    return features_list;
}


static PyObject * mdptetris_NewGame(PyObject *self, PyObject *args)
{
    int width;
    int height;
    int allow_overflow;
    char *pieces_filename = NULL;
    int *piece_sequence = NULL;
    PyObject *sequence = NULL;
    int seq_len;
    int i;
    if (!PyArg_ParseTuple(args, "iiis|O!", &width, &height, &allow_overflow, &pieces_filename,
                                                    &sequence))
        return NULL;

    if (sequence != NULL && PyList_Size(sequence) > 0)
    {
        seq_len = PyList_Size(sequence);
        piece_sequence = (int*)malloc(seq_len * sizeof(int));
        for (i = 0; i < seq_len; i++)
            piece_sequence[i] = PyInt_AsLong(PyList_GetItem(sequence,i));
    }

    if (tetris_game != NULL) {
        free_game(tetris_game);
        tetris_game = NULL;
    } else {
        MALLOC(original_features, FeaturePolicy);
        MALLOC(dellacherie_features, FeaturePolicy);
        create_feature_policy_original(width, original_features);
        create_feature_policy_dellacherie(width, dellacherie_features);
    }
    tetris_game = new_game(0, width, height, allow_overflow, pieces_filename, piece_sequence);


    return Py_BuildValue("");
}

static PyObject * mdptetris_Print(PyObject *self, PyObject *args)
{
    game_print(stdout, tetris_game);
    return Py_BuildValue("");
}

static PyObject * mdptetris_NumRotate(PyObject *self, PyObject *args)
{
    return Py_BuildValue("i", game_get_nb_possible_orientations(tetris_game));
}

static PyObject * mdptetris_NumPieces(PyObject *self, PyObject *args)
{
    return Py_BuildValue("i", game_get_nb_pieces(tetris_game));
}

static PyObject * mdptetris_NumColumns(PyObject *self, PyObject *args)
{
    int rotation;
     if (!PyArg_ParseTuple(args, "i", &rotation))
        return NULL;
    return Py_BuildValue("i", game_get_nb_possible_columns(tetris_game, rotation));
}

static PyObject * mdptetris_isGameOver(PyObject *self, PyObject *args)
{
    return Py_BuildValue("i", tetris_game->game_over);
}

static PyObject * mdptetris_CurrentPiece(PyObject *self, PyObject *args)
{
    return Py_BuildValue("i", game_get_current_piece(tetris_game));
}


static PyObject * mdptetris_DropPiece(PyObject *self, PyObject *args)
{
    int rotation;
    int column;
    int lines_cleared;
    Action action;

     if (!PyArg_ParseTuple(args, "ii", &rotation, &column))
        return NULL;

    action.orientation = rotation;
    action.column = column;
    lines_cleared = game_drop_piece(tetris_game, &action, 0);
    return Py_BuildValue("i", lines_cleared);
}

static PyMethodDef TetrisMethods[] = {
    {"tetris", mdptetris_NewGame, METH_VARARGS,"Start a new game of tetris."},
    {"reset_game", mdptetris_ResetGame, METH_VARARGS,"Reset the tetris game."},
    {"game_print",mdptetris_Print, METH_VARARGS,"Print the tetris game board."},
    {"num_pieces", mdptetris_NumPieces, METH_VARARGS,"Get the number of possible pieces."},
    {"num_rotate_actions", mdptetris_NumRotate, METH_VARARGS,"Get the number of rotation actions possible."},
    {"num_column_actions",mdptetris_NumColumns, METH_VARARGS,"Get the number of column actions possible."},
    {"drop_piece",mdptetris_DropPiece, METH_VARARGS,"Drop the current piece given a rotation action and column."},
    {"isgameover",mdptetris_isGameOver, METH_VARARGS,"Return if the game is over or not."},
    {"current_piece",mdptetris_CurrentPiece, METH_VARARGS,"Return int id of current game piece falling."},
    {"features_original",mdptetris_GetOriginalFeatures, METH_VARARGS,"Return original features."},
    {"features_dellacherie",mdptetris_GetDellacherieFeatures, METH_VARARGS,"Return Dellacherie features."},
    {"feature_ranges",mdptetris_FeaturesRanges, METH_VARARGS,"Return feature ranges for given feature set name."},
    {NULL, NULL, 0, NULL}
};


PyMODINIT_FUNC initmdptetris(void)
{
    PyObject *m;

    m = Py_InitModule("mdptetris", TetrisMethods);

    if (m == NULL)
        return;

}