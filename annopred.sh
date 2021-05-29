rm -rf test_output
mkdir test_output ## create dir for output files
rm -rf tmp_test
mkdir tmp_test ## create dir for temporary files
python AnnoPred.py\
  --sumstats=test_data/GWAS_sumstats.txt\
  --ref_gt=test_data/test\
  --val_gt=test_data/test\
  --coord_out=test_output/coord_out\
  --N_sample=69033\
  --annotation_flag="tier3"\
  --P=0.1\
  --local_ld_prefix=tmp_test/local_ld\
  --out=test_output/test\
  --temp_dir=tmp_test\
  --user_h2=test_data/user_h2_est.txt  
