#include <Python.h>
#include <numpy/arrayobject.h>
//#include <cmath.h>
#include <math.h>
#include <time.h>
#include "f2c.h"
#include "clapack.h"

#include <errno.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION


/*
# modify pinv: instead of x=A^-1B solve system of Ax=B and solve for x
    # i.e. cholesky decomposition done to solve system
# store A^-1 before hand (in look up table over curr_window_size)
    # or precompute and store lower cholesky
# possibly parallelize if possible
# curr_window_size range geyu thinks is ld_radius range
# make software more user friendly
# annopred_inf into c
# rob parallel branch of AnnoPred
    # contains the parallelization
    # conversion of printf to logging
    # lots of printf changes, even if each is small
    # think best way to merge
    # he suggests possible branch off his*/
/*
beta_hats - numpy array (numSnps,1)
pr_sigi - numpy array (numSnps,1)
h2 - double
n - int
reference_ld_mats - list (numSnps/ld_window_size+1) ith element numpy array square of size min(numSnps, (i+1)*ld_window_size)
ld_window_size - int
*/

static PyObject* annopred_inf(PyObject* self, PyObject* args) {
    /*
    infinitesimal model with snp-specific heritability derived from annotation
    used as the initial values for MCMC of non-infinitesimal model
    */
    int n, ld_window_size;
    double h2;
    PyArrayObject *pao_beta_hats, *pao_pr_sigi;
    PyListObject *plo_reference_ld_mats;
    
    PyObject *po_choleskyFunc,*po_solveFunc;
    PyObject *po_int;
    PyArrayObject *pao_A, *pao_b,*pao_updated_betas, *po_D,*pao_L,*pao_x,*pao_y;

    int start_i, stop_i, curr_window_size;

    if (!PyArg_ParseTuple(args, "O!O!diO!i",
        &PyArray_Type, &pao_beta_hats,
        &PyArray_Type,&pao_pr_sigi,
        &h2, 
        &n,
        &PyList_Type,&plo_reference_ld_mats,
        &ld_window_size)) return NULL;
        
    pao_beta_hats = (PyArrayObject *)PyArray_ContiguousFromObject((PyObject *)pao_beta_hats, NPY_DOUBLE, 1,1);
    pao_updated_betas=(PyArrayObject *)PyArray_FromDims(1, (int[]){pao_beta_hats->dimensions[0]}, NPY_DOUBLE);

    pao_pr_sigi = (PyArrayObject *)PyArray_ContiguousFromObject((PyObject *)pao_pr_sigi, NPY_DOUBLE, 1,1);

    po_choleskyFunc=PyDict_GetItemString(PyModule_GetDict(PyImport_AddModule("scipy.linalg")), "cholesky");
    po_solveFunc=PyDict_GetItemString(PyModule_GetDict(PyImport_AddModule("scipy.linalg")), "solve_triangular");
    
    curr_window_size=(int)fmin(ld_window_size,pao_beta_hats->dimensions[0]);
    pao_A=(PyArrayObject *)PyArray_FromDims(2, (int[]){curr_window_size,curr_window_size}, NPY_DOUBLE);
    pao_b=(PyArrayObject *)PyArray_FromDims(1, (int[]){curr_window_size}, NPY_DOUBLE);

    int i=0;
    for(int wi=0;wi<pao_beta_hats->dimensions[0];wi+=ld_window_size) {
        start_i = wi;
        stop_i = (int)fmin(pao_beta_hats->dimensions[0], wi + ld_window_size);
        curr_window_size = stop_i - start_i;

        //Li = 1.0/pr_sigi[start_i: stop_i]
        po_D = (PyArrayObject *)PyArray_ContiguousFromObject((PyObject *)PyList_GetItem((PyObject *)plo_reference_ld_mats,(Py_ssize_t) i), 
            NPY_DOUBLE,2,2);

        if (pao_A->dimensions[0]>curr_window_size) {
            Py_DECREF(pao_A);
            Py_DECREF(pao_b);
            pao_A=(PyArrayObject *)PyArray_FromDims(2, (int[]){curr_window_size,curr_window_size}, NPY_DOUBLE);
            pao_b=(PyArrayObject *)PyArray_FromDims(1, (int[]){curr_window_size}, NPY_DOUBLE);
        }            

        for (int pos =0;pos<curr_window_size;pos++) {
            *(double *)(pao_b->data + pos*pao_b->strides[0])=n*(*(double *)(pao_beta_hats->data + (start_i+pos)*pao_beta_hats->strides[0]));
        }
       
        for (int row =0;row<curr_window_size;row++) {
            for (int col=0;col<curr_window_size;col++) {
                if (row==col) {
                    *(double *)(pao_A->data + row*pao_A->strides[0]+col*pao_A->strides[1])=(pao_beta_hats->dimensions[0]/h2)+n *
                        *(double *)(po_D->data + row*po_D->strides[0]+col*po_D->strides[1]);
                } else {
                    *(double *)(pao_A->data + row*pao_A->strides[0]+col*pao_A->strides[1])=n* *(double *)(po_D->data + 
                        row*po_D->strides[0]+col*po_D->strides[1]);
                }
            }
        }

        Py_DECREF(po_D);
                
        //a, lower=False, overwrite_a=False, check_finite=True
        pao_L=(PyArrayObject *)PyObject_CallFunctionObjArgs(po_choleskyFunc, pao_A,Py_True,Py_False,Py_False, NULL);
        Py_INCREF(Py_True);
        Py_INCREF(Py_False);
        Py_INCREF(Py_False);        

        //a, b, trans=0, lower=False, unit_diagonal=False, overwrite_b=False, debug=None, check_finite=True
        po_int=PyInt_FromLong(0);
        pao_x=(PyArrayObject *)PyObject_CallFunctionObjArgs(po_solveFunc, pao_L,pao_b,po_int, Py_True, Py_False, Py_False, Py_False,
            Py_False,NULL);
        Py_INCREF(Py_True);
        Py_INCREF(Py_False);
        Py_INCREF(Py_False);
        Py_INCREF(Py_False);
        Py_INCREF(Py_False);
        Py_DECREF(po_int);

        //a, b, trans=0, lower=False, unit_diagonal=False, overwrite_b=False, debug=None, check_finite=True
        po_int=PyInt_FromLong(1);
        pao_y=(PyArrayObject *)PyObject_CallFunctionObjArgs(po_solveFunc, pao_L,pao_x,po_int, Py_True, Py_False, Py_False, Py_False,
            Py_False,NULL);
        Py_INCREF(Py_True);
        Py_INCREF(Py_False);
        Py_INCREF(Py_False);
        Py_INCREF(Py_False);
        Py_INCREF(Py_False);
        Py_DECREF(po_int);
        
        Py_DECREF(pao_x);
        Py_DECREF(pao_L);
        
        for (int pos=0;pos<curr_window_size;pos++) {
            *(double *)(pao_updated_betas->data + (pos+start_i)*pao_updated_betas->strides[0])=*(double *)(pao_y->data + 
                pos*pao_y->strides[0]);
        }

        Py_DECREF(pao_y);
        
        i++;
    }

    Py_DECREF(pao_beta_hats);
    Py_DECREF(pao_pr_sigi);
    Py_DECREF(plo_reference_ld_mats);
    Py_DECREF(pao_A);
    Py_DECREF(pao_b);
    
    return (PyObject *)pao_updated_betas;
}

/*
pao_beta_hats - numpy array (M,)
pao_Pi - numpy array (M,)
pao_Sigi2 - numpy array (M,)
pao_start_betas=None - (M,)
sig_12 - float 
h2=None - float
n=1000 - float
ld_radius=100 - float
num_iter=60 - float
burn_in=10 - float
zero_jump_prob=0.05 - float
po_ld_dict=None - dict of keys 0 to M-1
*/
static PyObject* non_infinitesimal_mcmc(PyObject* self, PyObject* args) {
    PyObject *po_ld_dict;
    PyObject *po_long;
    PyObject *po_randomFunc,*po_randomNormalFunc;
    PyObject *po_randomNormalLoc,*po_randomNormalScale;
    PyObject *po_randomStr,*po_normalStr;
    
    PyArrayObject *pao_avg_betas, *pao_curr_betas, *pao_curr_post_means,*pao_D_i,*pao_rand_ps;
    PyArrayObject *pao_beta_hats, *pao_Pi, *pao_Sigi2,*pao_start_betas;

    double *local_betas ;  /* could make this any variable type */
    double hdmp, hdmpn, hdmp_hdmpn, c_const, d_const,postp;
    double b2,alpha,h2_est,res_beta_hat_i,d_const_b2_exp,numerator,proposed_beta;
    double sig_12, h2, n, ld_radius, num_iter, burn_in, zero_jump_prob;
    int start_i,stop_i,focal_i;
    
    if (!PyArg_ParseTuple(args, "O!O!O!O!dddddddO!", 
        &PyArray_Type, &pao_beta_hats,&PyArray_Type, &pao_Pi,&PyArray_Type,&pao_Sigi2,&PyArray_Type, &pao_start_betas,
        &sig_12,&h2,&n,&ld_radius,&num_iter,&burn_in,&zero_jump_prob,
        &PyDict_Type,&po_ld_dict)) return NULL;

    // convert numpy array arguments to contiguous arrays
    pao_beta_hats = (PyArrayObject *)PyArray_ContiguousFromObject((PyObject *)pao_beta_hats, NPY_DOUBLE, 1,1);
    pao_Pi = (PyArrayObject *)PyArray_ContiguousFromObject((PyObject *)pao_Pi, NPY_DOUBLE, 1,1);
    pao_Sigi2 = (PyArrayObject *)PyArray_ContiguousFromObject((PyObject *)pao_Sigi2, NPY_DOUBLE, 1,1);
    pao_start_betas = (PyArrayObject *)PyArray_ContiguousFromObject((PyObject *)pao_start_betas, NPY_DOUBLE, 1,1);
    
    // find the number of snps and save in local variable
    int numSnps=pao_beta_hats->dimensions[0];
    int snpSize[] = {numSnps};

    // load the functions numpy.random.random into po_randomFunc
    po_randomStr=PyString_FromString("random");
    po_randomFunc=PyDict_GetItem(PyModule_GetDict(PyImport_AddModule("numpy.random")), po_randomStr);   
    Py_DECREF(po_randomStr);

    // load the function numpy.random.normal into po_randomNormalFunc
    po_normalStr = PyString_FromString("normal");
    po_randomNormalFunc=PyDict_GetItem(PyModule_GetDict(PyImport_AddModule("numpy.random")), po_normalStr);
    Py_DECREF(po_normalStr);
    
    // create new numpy 1-d arrays to hold results
    pao_avg_betas=(PyArrayObject *)PyArray_FromDims(1, snpSize, NPY_DOUBLE);
    pao_curr_betas=(PyArrayObject *)PyArray_FromDims(1, snpSize, NPY_DOUBLE);
    pao_curr_post_means=(PyArrayObject *)PyArray_FromDims(1, snpSize, NPY_DOUBLE);
    
    // initialize curr_betas, curr_post_means and avg_betas
    for(int snp=0;snp<numSnps;snp++) {
        /* do something with the data at dptr */
        *(double *)(pao_curr_betas->data + snp*pao_curr_betas->strides[0])=*(double *)(pao_start_betas->data + 
            snp*pao_start_betas->strides[0]);
        *(double *)(pao_curr_post_means->data + snp*pao_curr_post_means->strides[0])=0;
        *(double *)(pao_avg_betas->data + snp*pao_avg_betas->strides[0])=0;
    }

    // main loop over iterations of mcmc
    for (int iter=0;iter<num_iter;iter++) {
        h2_est=0;
        for(int snp=0;snp<numSnps;snp++) {
            h2_est+=pow(*(double *)(pao_curr_betas->data + snp*pao_curr_betas->strides[0]),2);
        }        
        h2_est=fmax(h2_est,0.00001);

        alpha = fmin(fmin(1-zero_jump_prob, 1.0 / h2_est), (h2 + 1 / sqrt(n)) / h2_est);      
        
        // generate a vector of random iid uniforms
        po_long=PyLong_FromLong(numSnps);
        pao_rand_ps=(PyArrayObject *)PyObject_CallFunctionObjArgs(po_randomFunc, po_long, NULL);
        Py_DECREF(po_long);

        
        // sub loop over snps
        for (int snp=0;snp<numSnps;snp++){
            if (*(double *)(pao_Sigi2->data + snp*pao_Sigi2->strides[0])==0) {
                *(double *)(pao_curr_post_means->data + snp*pao_curr_post_means->strides[0]) = 0;
                *(double *)(pao_curr_betas->data + snp*pao_curr_betas->strides[0]) = 0;
            } else {
                hdmp=(*(double *)(pao_Sigi2->data + snp*pao_Sigi2->strides[0]))/(*(double *)(pao_Pi->data + snp*pao_Pi->strides[0])); // (h2 / Mp)
                hdmpn = hdmp + sig_12; // 1.0 / n
                hdmp_hdmpn = (hdmp / hdmpn);
                c_const= (*(double *)(pao_Pi->data + snp*pao_Pi->strides[0])) / sqrt(hdmpn);
                d_const= (1 - *(double *)(pao_Pi->data + snp*pao_Pi->strides[0]))/sqrt(sig_12);
    
                start_i = (int)fmax(0, snp - ld_radius);
                focal_i = (int)fmin(ld_radius, snp);
                stop_i = (int)fmin(numSnps, snp + ld_radius + 1);
                
                // Local LD matrix
                po_long=PyLong_FromLong(snp);
                pao_D_i=(PyArrayObject *)PyArray_ContiguousFromObject(PyDict_GetItem(po_ld_dict, po_long),PyArray_DOUBLE, 1,1);
                Py_DECREF(po_long);
                
                // Local (most recently updated) effect estimates

                local_betas=(double *)malloc(sizeof(double)*(stop_i-start_i));
                for (int s=0;s<(stop_i-start_i);s++){
                    local_betas[s] = *(double *)(pao_curr_betas->data + (start_i+s)*pao_curr_betas->strides[0]);
                }
                
                // Calculate the local posterior mean, used when sampling.
                local_betas[focal_i]=0;
                
                res_beta_hat_i=(*(double *)(pao_beta_hats->data + snp*pao_beta_hats->strides[0]));
                for (int s=0;s<=(stop_i-start_i);s++){
                    res_beta_hat_i-=(*(double *)(pao_D_i->data + s*pao_D_i->strides[0])*local_betas[s]);
                }
                free(local_betas);
                Py_DECREF(pao_D_i);
                
                b2=pow(res_beta_hat_i,2);
        
                d_const_b2_exp = d_const * exp(-b2 / (2.0*sig_12));
                numerator = c_const * exp(-b2 / (2.0 * hdmpn));
                if (numerator == 0) {
                    postp = 0;
                } else {
                    postp = numerator / (numerator + d_const_b2_exp);
                }
                (*(double *)(pao_curr_post_means->data + snp*pao_curr_post_means->strides[0]))=hdmp_hdmpn * postp * res_beta_hat_i;
        
                if ((*(double *)(pao_rand_ps->data + snp*pao_rand_ps->strides[0])) < postp * alpha) {
                    // Sample from the posterior Gaussian dist.
                    po_randomNormalLoc=PyFloat_FromDouble(0);
                    po_randomNormalScale=PyFloat_FromDouble(hdmp_hdmpn * sig_12);
                    proposed_beta=PyFloat_AsDouble(PyObject_CallFunctionObjArgs(po_randomNormalFunc, 
                        po_randomNormalLoc,po_randomNormalScale, NULL)) + hdmp_hdmpn * res_beta_hat_i;
                    Py_DECREF(po_randomNormalLoc);
                    Py_DECREF(po_randomNormalScale); 
                } else {
                    // Sample 0
                    proposed_beta = 0;
                }
                
                (*(double *)(pao_curr_betas->data + snp*pao_curr_betas->strides[0]))=proposed_beta;                
            }            
        } 
        
        Py_DECREF(pao_rand_ps);  // relese random vector of uniforms
           
        // update avg_betas with results once burn in number of iterations have passed
        if (iter >= burn_in) {
            for (int snp=0;snp<numSnps;snp++) {
                (*(double *)(pao_avg_betas->data + snp*pao_avg_betas->strides[0]))+=(*(double *)(pao_curr_post_means->data + 
                    snp*pao_curr_post_means->strides[0]));
            }
        }
    }
    
    // divide avg_betas by number of contributing iterations    
    for (int snp=0;snp<numSnps;snp++) {
        (*(double *)(pao_avg_betas->data + snp*pao_avg_betas->strides[0]))/=(double)(num_iter-burn_in);
    }

    // release the numpy references to parameter arguments
    Py_DECREF(pao_beta_hats);
    Py_DECREF(pao_Pi);
    Py_DECREF(pao_Sigi2);
    Py_DECREF(pao_start_betas);
    
    // release the number references to arrays created from scratch in this function
    Py_DECREF(pao_curr_betas);
    Py_DECREF(pao_curr_post_means);
    
    return (PyObject *)pao_avg_betas;
}

static PyMethodDef CAnnoPred_funcs[] = {
    {"non_infinitesimal_mcmc", non_infinitesimal_mcmc, METH_VARARGS, "compute non_infinitesimal_mcmc"},
    {"annopred_inf", annopred_inf, METH_KEYWORDS, "compute annopred_inf"},
    { NULL, NULL, 0, NULL }
};

PyMODINIT_FUNC initCAnnoPred(void) {
    (void) Py_InitModule("CAnnoPred", CAnnoPred_funcs);  
    import_array();
}

