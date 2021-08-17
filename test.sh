#!/bin/bash

#SBATCH -c 16
#SBATCH -t 1-0
#SBATCH -p bigmem -C cascadelake
#SBATCH --mem=750g
#SBATCH --output=CAnnoPred.out
#SBATCH --error=CAnnoPred.err

#rm -rf test_output
#mkdir test_output ## create dir for output files
#rm -rf tmp_test
#mkdir tmp_test ## create dir for temporary files

#module load miniconda
#conda activate /home/ta459/ldsc

#python -m cProfile -o annopred4.prof AnnoPred.py\
#/home/ta459/ldsc/bin/mprof run --include-children

#python -m memory_profiler 
# -m memprof --plot 
python AnnoPred.py \
  --sumstats=Basic_EUR_1000G_IBD_2015.txt\
  --ref_gt=1000G_mac5eur\
  --val_gt=1000G_mac5eur\
  --coord_out=test_output/coord_out\
  --N_sample=69033\
  --annotation_flag="tier3"\
  --P=0.1\
  --local_ld_prefix=tmp_test/local_ld\
  --out=test_output/test\
  --temp_dir=tmp_test
  
#python AnnoPred.py \
#    --sumstats=/gpfs/ysm/pi/zhao-data/jh2875/cancer_prs/data/annopred/gwas/breast.txt \
#    --ref_gt=/gpfs/ysm/pi/zhao/yy496/PRS/GWAS_1000G_mac5eur_mapping/1000G_mac5eur \
#    --val_gt=/gpfs/ysm/pi/zhao/yy496/PRS/GWAS_1000G_mac5eur_mapping/1000G_mac5eur \
#    --coord_out=tmp_test/coord_out \
#    --N_sample=247173 \
#    --annotation_flag="tier0" \
#    --P=1 \
#    --local_ld_prefix=temp/local_ld \
#    --out=test_output \
#    --temp_dir=tmp_test \
#    --ld_radius=1500 

# cd /gpfs/ysm/pi/zhao/jh2875/AnnoPred/; python ~/CAnnoPred/AnnoPred.py --sumstats=/gpfs/ysm/pi/zhao-data/jh2875/cancer_prs/data/annopred/gwas/breast.txt --ref_gt=/gpfs/ysm/pi/zhao/yy496/PRS/GWAS_1000G_mac5eur_mapping/1000G_mac5eur --val_gt=/gpfs/ysm/pi/zhao/yy496/PRS/GWAS_1000G_mac5eur_mapping/1000G_mac5eur --coord_out=/gpfs/ysm/pi/zhao-data/jh2875/cancer_prs/data/annopred/temp/coord_out --N_sample=247173 --annotation_flag="tier0" --P=1 --local_ld_prefix=/gpfs/ysm/pi/zhao-data/jh2875/cancer_prs/data/annopred/temp/local_ld --out=/gpfs/ysm/pi/zhao-data/jh2875/cancer_prs/data/annopred/score/breast --temp_dir=/gpfs/ysm/pi/zhao-data/jh2875/cancer_prs/data/annopred/temp --ld_radius=1500

#merge into annopred robparallel, then branch off of that
# run a compute node
# rob will send the profile script
# email yixuan mepred
# do memory analysis, use sparse matrix, find the largest memory used
# find object that is too big