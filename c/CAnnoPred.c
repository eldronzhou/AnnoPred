#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <math.h>
#include <time.h>

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

#define min(a,b) (((a)<(b))?(a):(b))
#define max(a,b) (((a)>(b))?(a):(b))

/*
    po_float=PyObject_CallFunctionObjArgs(PyDict_GetItemString(PyModule_GetDict(PyImport_AddModule("time")), "time"), NULL);
    PyObject_Print(po_float,stdout,Py_PRINT_RAW);
    Py_DECREF(po_float);
    printf("\n");
    
*/

static PyObject* get_LDpred_ld_tables(PyObject* self, PyObject* args) {
    int i_ld_radius,i_window_size,i_m, i_n, i_temp, i_start, i_stop;
    float f_sigma, f_mu;
    double d_avg_ld_score, d_temp;
    PyObject *pao_snps, *pao_snp, *pao_arg_raw_snps,*pao_arg_snp_stds,*pao_arg_snp_means,*pao_X, *pao_prod;
    PyObject *po_float;
    PyObject *pdo_out;
    PyObject *plo_ref_ld_matrices, *plo_ld_matrices, *plo_ld_scores;
    
    npy_intp dim2[2], dim1[2];
        
    char *pc_snps_0, *pc_snps_1, *pc_arg_raw_snps_0, *pc_arg_raw_snps_1, *pc_arg_snp_means_0, *pc_arg_snp_stds_0;

    if (!PyArg_ParseTuple(args, "O!O!O!ii",&PyArray_Type, &pao_arg_raw_snps,&PyArray_Type, &pao_arg_snp_means,&PyArray_Type, 
                          &pao_arg_snp_stds,&i_ld_radius,&i_window_size)) return NULL;

    pdo_out=PyDict_New();
    po_float=PyObject_CallFunctionObjArgs(PyDict_GetItemString(PyModule_GetDict(PyImport_AddModule("time")), "time"), NULL);
    PyObject_Print(po_float,stdout,Py_PRINT_RAW);
    Py_DECREF(po_float);
    printf("\n");
    
    i_m=0;
    pc_arg_snp_stds_0=PyArray_BYTES((PyArrayObject*)pao_arg_snp_stds);
    for(int row=0;row<PyArray_DIMS((PyArrayObject*)pao_arg_snp_stds)[0];row++) {
        i_m+=(((float *)pc_arg_snp_stds_0)[0]>0)?1:0;
        pc_arg_snp_stds_0+=PyArray_STRIDES((PyArrayObject*)pao_arg_snp_stds)[0];
    }
    
    i_n=PyArray_DIMS((PyArrayObject*)pao_arg_raw_snps)[1];
    
    dim2[0]=i_m;
    dim2[1]=i_n;
    
    pao_snps=PyArray_SimpleNew(2, dim2, NPY_DOUBLE);

    pc_arg_raw_snps_0=PyArray_BYTES((PyArrayObject*)pao_arg_raw_snps);
    pc_arg_snp_stds_0=PyArray_BYTES((PyArrayObject*)pao_arg_snp_stds);
    pc_arg_snp_means_0=PyArray_BYTES((PyArrayObject*)pao_arg_snp_means);
    pc_snps_0=PyArray_BYTES((PyArrayObject*)pao_snps);
    
    for(int row=0;row<PyArray_DIMS((PyArrayObject*)pao_arg_raw_snps)[0];row++) {
        if (((float *)pc_arg_snp_stds_0)[0]>0) {                        
            f_mu=((float *)pc_arg_snp_means_0)[0];
            f_sigma=((float *)pc_arg_snp_stds_0)[0]*sqrt(i_n);
            
            pc_arg_raw_snps_1=pc_arg_raw_snps_0;
            pc_snps_1=pc_snps_0;
            for(int col=0;col<i_n;col++){ 
                ((double *)pc_snps_1)[0]=(((signed char *)pc_arg_raw_snps_1)[0]-f_mu)/f_sigma;
                pc_arg_raw_snps_1+=PyArray_STRIDES((PyArrayObject*)pao_arg_raw_snps)[1];
                pc_snps_1+=PyArray_STRIDES((PyArrayObject*)pao_snps)[1];
            }
                        
            pc_snps_0+=PyArray_STRIDES((PyArrayObject *)pao_snps)[0];
        }
        
        pc_arg_snp_stds_0+=PyArray_STRIDES((PyArrayObject*)pao_arg_snp_stds)[0];
        pc_arg_snp_means_0+=PyArray_STRIDES((PyArrayObject*)pao_arg_snp_means)[0];
        pc_arg_raw_snps_0+=PyArray_STRIDES((PyArrayObject*)pao_arg_raw_snps)[0];
    }
    po_float=PyObject_CallFunctionObjArgs(PyDict_GetItemString(PyModule_GetDict(PyImport_AddModule("time")), "time"), NULL);
    PyObject_Print(po_float,stdout,Py_PRINT_RAW);
    Py_DECREF(po_float);
    printf("\n");
    
    plo_ref_ld_matrices=PyList_New(1+(npy_intp)floor((i_m-1.0)/max(1,i_window_size)));
    plo_ld_scores=PyList_New(i_m);
    plo_ld_matrices=PyList_New(i_m);
    
    d_avg_ld_score=0;
    
    i_temp=0;
    pc_snps_0=PyArray_BYTES((PyArrayObject*)pao_snps);
    pc_snps_1=pc_snps_0;
    dim1[0]=1;
    dim1[1]=i_n;
    for (int ind=0;ind<i_m;ind++){ 
        i_start=max(0,ind - i_ld_radius);
        i_stop=min(i_m, ind + i_ld_radius + 1);

        pao_snp=PyArray_SimpleNewFromData(2, dim1, NPY_DOUBLE, pc_snps_0);
        
        dim2[0]=(i_stop-i_start); 
        
        if (ind>i_ld_radius) pc_snps_1+=PyArray_STRIDES((PyArrayObject*)pao_snps)[0];
        pao_X=PyArray_SimpleNewFromData(2, dim2, NPY_DOUBLE, pc_snps_1);
        
        pao_prod=PyArray_MatrixProduct(pao_X,PyArray_Transpose((PyArrayObject *)pao_snp,NULL));
        
        PyArray_ScalarAsCtype(PyArray_Return((PyArrayObject*)PyArray_InnerProduct(pao_prod,pao_prod)),&d_temp);
        
        d_temp=(d_temp*(i_n-1)-PyArray_DIMS((PyArrayObject*)pao_X)[0])/(i_n-2);
        
        d_avg_ld_score+=d_temp;
                            
        PyList_SetItem(plo_ld_matrices, i_temp, pao_prod);
        PyList_SetItem(plo_ld_scores, i_temp, PyFloat_FromDouble(d_temp));
        
        i_temp++;
        pc_snps_0+=PyArray_STRIDES((PyArrayObject*)pao_snps)[0];
    }        
    po_float=PyObject_CallFunctionObjArgs(PyDict_GetItemString(PyModule_GetDict(PyImport_AddModule("time")), "time"), NULL);
    PyObject_Print(po_float,stdout,Py_PRINT_RAW);
    Py_DECREF(po_float);
    printf("\n");
    
    Py_INCREF(Py_None);
    return Py_None;
    
    i_temp=0;
    pc_snps_0=PyArray_BYTES((PyArrayObject*)pao_snps);    
    for (int ind=0;ind<i_m;ind+=i_window_size){
        dim2[0]=min(i_m, ind + i_window_size)-ind;
        
        pao_X=PyArray_SimpleNewFromData(2, dim2, NPY_DOUBLE, pc_snps_0);

        pao_prod=PyArray_MatrixProduct(pao_X,PyArray_Transpose((PyArrayObject *)pao_X,NULL));

        PyList_SetItem(plo_ref_ld_matrices, i_temp, pao_prod);
        
        i_temp++;
        pc_snps_0+=i_window_size*PyArray_STRIDES((PyArrayObject*)pao_snps)[0];
    }        
    
    po_float=PyObject_CallFunctionObjArgs(PyDict_GetItemString(PyModule_GetDict(PyImport_AddModule("time")), "time"), NULL);
    PyObject_Print(po_float,stdout,Py_PRINT_RAW);
    Py_DECREF(po_float);
    printf("\n");

    PyDict_SetItemString(pdo_out, "ref_ld_matrices", plo_ref_ld_matrices);
    PyDict_SetItemString(pdo_out, "ld_scores", plo_ld_scores);
    PyDict_SetItemString(pdo_out, "ld_matrices", plo_ld_matrices);
    
    po_float=PyFloat_FromDouble(d_avg_ld_score);
    PyDict_SetItemString(pdo_out, "avg_ld_score", po_float);
    Py_DECREF(po_float);
    
    Py_DECREF(plo_ld_scores);
    Py_DECREF(plo_ld_matrices);
    Py_DECREF(plo_ref_ld_matrices);

    return pdo_out;
}

/*
beta_hats - numpy array (numSnps,1)
pr_sigi - numpy array (numSnps,1)
h2 - double
n - int
reference_ld_mats - list (numSnps/ld_window_size+1) ith element numpy array square of size min(numSnps, (i+1)*ld_window_size)
ld_window_size - int
*/
static PyObject* annopred_inf(PyObject* self, PyObject* args) {
    // infinitesimal model with snp-specific heritability derived from annotation
    // used as the initial values for MCMC of non-infinitesimal model
    
    int n, ld_window_size;
    double h2;
    PyArrayObject *pao_beta_hats, *pao_pr_sigi;
    PyListObject *plo_reference_ld_mats;
    
    PyObject *po_choleskyFunc,*po_solveFunc;
    PyObject *po_int;
    PyArrayObject *pao_A, *pao_b,*pao_updated_betas, *po_D,*pao_L,*pao_x,*pao_y;
    
    npy_intp dim2[2], dim1[1];

    int start_i, stop_i, curr_window_size;

    if (!PyArg_ParseTuple(args, "O!O!diO!i",
        &PyArray_Type, &pao_beta_hats,
        &PyArray_Type,&pao_pr_sigi,
        &h2, 
        &n,
        &PyList_Type,&plo_reference_ld_mats,
        &ld_window_size)) return NULL;
        
    Py_INCREF(Py_None);
    return Py_None;
    pao_beta_hats = (PyArrayObject *)PyArray_ContiguousFromObject((PyObject *)pao_beta_hats, NPY_DOUBLE, 1,1);
    dim1[0]=PyArray_DIMS(pao_beta_hats)[0];
    pao_updated_betas=(PyArrayObject *)PyArray_SimpleNew(1, dim1, NPY_DOUBLE);

    pao_pr_sigi = (PyArrayObject *)PyArray_ContiguousFromObject((PyObject *)pao_pr_sigi, NPY_DOUBLE, 1,1);

    po_choleskyFunc=PyDict_GetItemString(PyModule_GetDict(PyImport_AddModule("scipy.linalg")), "cholesky");
    po_solveFunc=PyDict_GetItemString(PyModule_GetDict(PyImport_AddModule("scipy.linalg")), "solve_triangular");
    
    curr_window_size=(int)fmin(ld_window_size,PyArray_DIMS(pao_beta_hats)[0]);
    
    dim2[0]=curr_window_size;
    dim2[1]=curr_window_size;
    pao_A=(PyArrayObject *)PyArray_SimpleNew(2, dim2, NPY_DOUBLE);
    
    dim1[0]=curr_window_size;
    pao_b=(PyArrayObject *)PyArray_SimpleNew(1, dim1, NPY_DOUBLE);

    int i=0;
    for(int wi=0;wi<PyArray_DIMS(pao_beta_hats)[0];wi+=ld_window_size) {
        start_i = wi;
        stop_i = (int)fmin(PyArray_DIMS(pao_beta_hats)[0], wi + ld_window_size);
        curr_window_size = stop_i - start_i;

        //Li = 1.0/pr_sigi[start_i: stop_i]
        po_D = (PyArrayObject *)PyArray_ContiguousFromObject((PyObject *)PyList_GetItem((PyObject *)plo_reference_ld_mats,(Py_ssize_t) i), 
            NPY_DOUBLE,2,2);

        if (PyArray_DIMS(pao_A)[0]>curr_window_size) {
            Py_DECREF(pao_A);
            Py_DECREF(pao_b);
            
            dim2[0]=curr_window_size;
            dim2[1]=curr_window_size;
            pao_A=(PyArrayObject *)PyArray_SimpleNew(2, dim2, NPY_DOUBLE);
            
            dim1[0]=curr_window_size;
            pao_b=(PyArrayObject *)PyArray_SimpleNew(1, dim1, NPY_DOUBLE);
        }            

        for (int pos =0;pos<curr_window_size;pos++) {
            ((double *)PyArray_GETPTR1(pao_b,pos))[0]=n*((double *)PyArray_GETPTR1(pao_beta_hats,start_i+pos))[0];
        }
       
        for (int row =0;row<curr_window_size;row++) {
            for (int col=0;col<curr_window_size;col++) {
                if (row==col) {
                    ((double *)PyArray_GETPTR2(pao_A,row,col))[0]=(PyArray_DIMS(pao_beta_hats)[0]/h2)+n *
                        ((double *)PyArray_GETPTR2(po_D,row,col))[0];
                } else {
                    ((double *)PyArray_GETPTR2(pao_A,row,col))[0]=n* ((double *)PyArray_GETPTR2(po_D,row,col))[0];
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
        po_int=PyLong_FromLong(0);
        pao_x=(PyArrayObject *)PyObject_CallFunctionObjArgs(po_solveFunc, pao_L,pao_b,po_int, Py_True, Py_False, Py_False, Py_False,
            Py_False,NULL);
        Py_INCREF(Py_True);
        Py_INCREF(Py_False);
        Py_INCREF(Py_False);
        Py_INCREF(Py_False);
        Py_INCREF(Py_False);
        Py_DECREF(po_int);

        //a, b, trans=0, lower=False, unit_diagonal=False, overwrite_b=False, debug=None, check_finite=True
        po_int=PyLong_FromLong(1);
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
            ((double *)PyArray_GETPTR1(pao_updated_betas,pos+start_i))[0]=((double *)PyArray_GETPTR1(pao_y,pos))[0];
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
    
    PyArrayObject *pao_avg_betas, *pao_curr_betas, *pao_curr_post_means,*pao_D_i,*pao_rand_ps;
    PyArrayObject *pao_beta_hats, *pao_Pi, *pao_Sigi2,*pao_start_betas;

    double *local_betas ;  // could make this any variable type 
    double hdmp, hdmpn, hdmp_hdmpn, c_const, d_const,postp;
    double b2,alpha,h2_est,res_beta_hat_i,d_const_b2_exp,numerator,proposed_beta;
    double sig_12, h2, n, ld_radius, num_iter, burn_in, zero_jump_prob;
    int start_i,stop_i,focal_i;
    
    npy_intp dim1[1];

    if (!PyArg_ParseTuple(args, "O!O!O!O!dddddddO!", 
        &PyArray_Type, &pao_beta_hats,&PyArray_Type, &pao_Pi,&PyArray_Type,&pao_Sigi2,&PyArray_Type, &pao_start_betas,
        &sig_12,&h2,&n,&ld_radius,&num_iter,&burn_in,&zero_jump_prob,
        &PyDict_Type,&po_ld_dict)) return NULL;

    // convert numpy array arguments to contiguous arrays
    //Py_INCREF(Py_None);
    //return Py_None;
    pao_beta_hats = (PyArrayObject *)PyArray_ContiguousFromObject((PyObject *)pao_beta_hats, NPY_DOUBLE, 1,1);
    pao_Pi = (PyArrayObject *)PyArray_ContiguousFromObject((PyObject *)pao_Pi, NPY_DOUBLE, 1,1);
    pao_Sigi2 = (PyArrayObject *)PyArray_ContiguousFromObject((PyObject *)pao_Sigi2, NPY_DOUBLE, 1,1);
    pao_start_betas = (PyArrayObject *)PyArray_ContiguousFromObject((PyObject *)pao_start_betas, NPY_DOUBLE, 1,1);
    
    // find the number of snps and save in local variable
    int numSnps=PyArray_DIMS(pao_beta_hats)[0];
    int snpSize[] = {numSnps};

    //Py_INCREF(Py_None);
    //return Py_None;
    // load the functions numpy.random.random into po_randomFunc
    po_randomFunc=PyDict_GetItemString(PyModule_GetDict(PyImport_AddModule("numpy.random")), "random");

    // load the function numpy.random.normal into po_randomNormalFunc
    po_randomNormalFunc=PyDict_GetItemString(PyModule_GetDict(PyImport_AddModule("numpy.random")), "normal");
    
    // create new numpy 1-d arrays to hold results
    dim1[0]=(npy_intp)snpSize;
    pao_avg_betas=(PyArrayObject *)PyArray_SimpleNew(1, dim1, NPY_DOUBLE);
    pao_curr_betas=(PyArrayObject *)PyArray_SimpleNew(1, dim1, NPY_DOUBLE);
    pao_curr_post_means=(PyArrayObject *)PyArray_SimpleNew(1, dim1, NPY_DOUBLE);
    
    // initialize curr_betas, curr_post_means and avg_betas
    for(int snp=0;snp<numSnps;snp++) {
        // do something with the data at dptr 
        ((double *)PyArray_GETPTR1(pao_curr_betas,snp))[0]=((double *)PyArray_GETPTR1(pao_start_betas,snp))[0];
        ((double *)PyArray_GETPTR1(pao_curr_post_means,snp))[0]=0;
        ((double *)PyArray_GETPTR1(pao_avg_betas,snp))[0]=0;
    }

    //Py_INCREF(Py_None);
    //return Py_None;
    // main loop over iterations of mcmc
    for (int iter=0;iter<num_iter;iter++) {
        h2_est=0;
        for(int snp=0;snp<numSnps;snp++) {
            h2_est+=pow(((double *)PyArray_GETPTR1(pao_curr_betas,snp))[0],2);
        }        
        h2_est=fmax(h2_est,0.00001);

        alpha = fmin(fmin(1-zero_jump_prob, 1.0 / h2_est), (h2 + 1 / sqrt(n)) / h2_est);      
        
        // generate a vector of random iid uniforms
        po_long=PyLong_FromLong(numSnps);
        pao_rand_ps=(PyArrayObject *)PyObject_CallFunctionObjArgs(po_randomFunc, po_long, NULL);
        Py_DECREF(po_long);

        //Py_INCREF(Py_None);
        //return Py_None;

        // sub loop over snps
        for (int snp=0;snp<numSnps;snp++){
            if (((double *)PyArray_GETPTR1(pao_Sigi2,snp))[0]==0) {
                ((double *)PyArray_GETPTR1(pao_curr_post_means,snp))[0] = 0;
                ((double *)PyArray_GETPTR1(pao_curr_betas,snp))[0] = 0;
            } else {
                hdmp=((double *)PyArray_GETPTR1(pao_Sigi2,snp))[0]/((double *)PyArray_GETPTR1(pao_Pi,snp))[0]; // (h2 / Mp)
                hdmpn = hdmp + sig_12; // 1.0 / n
                hdmp_hdmpn = (hdmp / hdmpn);
                c_const= ((double *)PyArray_GETPTR1(pao_Pi,snp))[0] / sqrt(hdmpn);
                d_const= (1 - ((double *)PyArray_GETPTR1(pao_Pi,snp))[0])/sqrt(sig_12);
    
                start_i = (int)fmax(0, snp - ld_radius);
                focal_i = (int)fmin(ld_radius, snp);
                stop_i = (int)fmin(numSnps, snp + ld_radius + 1);
                
                // Local LD matrix
                po_long=PyLong_FromLong(snp);
                pao_D_i=(PyArrayObject *)PyArray_ContiguousFromObject(PyDict_GetItem(po_ld_dict, po_long),NPY_DOUBLE, 1,1);
                Py_DECREF(po_long);
                
                // Local (most recently updated) effect estimates

                local_betas=(double *)malloc(sizeof(double)*(stop_i-start_i));
                for (int s=0;s<(stop_i-start_i);s++){
                    local_betas[s] = ((double *)PyArray_GETPTR1(pao_curr_betas,start_i+s))[0];
                }
                //Py_INCREF(Py_None);
                //return Py_None;

                // Calculate the local posterior mean, used when sampling.
                local_betas[focal_i]=0;
                
                res_beta_hat_i=((double *)PyArray_GETPTR1(pao_beta_hats,snp))[0];
                for (int s=0;s<=(stop_i-start_i);s++){
                    res_beta_hat_i-=((double *)PyArray_GETPTR1(pao_D_i,s))[0]*local_betas[s];
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
                ((double *)PyArray_GETPTR1(pao_curr_post_means,snp))[0]=hdmp_hdmpn * postp * res_beta_hat_i;
                //Py_INCREF(Py_None);
                //return Py_None;

                if (((double *)PyArray_GETPTR1(pao_rand_ps,snp))[0] < postp * alpha) {
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
                
                ((double *)PyArray_GETPTR1(pao_curr_betas,snp))[0]=proposed_beta;                
            }            
        } 
        Py_INCREF(Py_None);
        return Py_None;

        Py_DECREF(pao_rand_ps);  // relese random vector of uniforms
           
        // update avg_betas with results once burn in number of iterations have passed
        if (iter >= burn_in) {
            for (int snp=0;snp<numSnps;snp++) {
                ((double *)PyArray_GETPTR1(pao_avg_betas,snp))[0]+=((double *)PyArray_GETPTR1(pao_curr_post_means,snp))[0];
            }
        }
    }
    
    // divide avg_betas by number of contributing iterations    
    for (int snp=0;snp<numSnps;snp++) {
        ((double *)PyArray_GETPTR1(pao_avg_betas,snp))[0]/=(double)(num_iter-burn_in);
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
    {"annopred_inf", annopred_inf, METH_VARARGS, "compute annopred_inf"},
    {"get_LDpred_ld_tables", get_LDpred_ld_tables, METH_VARARGS, "compute get_LDpred_ld_tables"},
    { NULL, NULL, 0, NULL }
};

static struct PyModuleDef cannopredmodule = {
    PyModuleDef_HEAD_INIT,
    "CAnnoPred",   //name of module 
    NULL,          // module documentation, may be NULL 
    -1,            // size of per-interpreter state of the module, or -1 if the module keeps state in global variables. 
    CAnnoPred_funcs
};

PyMODINIT_FUNC PyInit_CAnnoPred(void) {
    import_array();
    return PyModule_Create(&cannopredmodule);  
}
