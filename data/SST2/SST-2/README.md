# Data Processing

## Traing SPM model

```shell
awk 'FNR>1' train.tsv | cut -f 1  > senetence.txt
spm_train --input=senetence.txt --model_prefix=lstm --character_coverage=1.0 --pad_id 3
```

