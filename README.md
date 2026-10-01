## ADNI1 Dataset Folders:
    |
    -- ADNI1-Complete 1Yr 3T (119 Subjects)
    -- ADNI1_Annual 2 Yr 3T  (89 Subjects)
    -- ADNI1_Complete 2Yr 3T (86 Subjects)
    -- ADNI1-Complete 3Yr 3T (60 Subjects)
    
## Across all four files combined:
    * There are 594 unique Image Data IDs in total.
    * There are 151 Image Data IDs common to all four datasets!

        Among the 594 unique Image Data IDs across all four datasets:
        195 belong to the CN (Cognitively Normal) group,
        301 belong to the MCI (Mild Cognitive Impairment) group, and
        98 belong to the AD (Alzheimer’s Disease) group
        
    
# In adni_train_torch.py, change line 27 from:
from adni_dataloader import ADNIDataLoader # Subject-level splitting
# To:
from adni_dataloader_scan_level import ADNIDataLoaderScanLevel # Scan-level splitting

# And change line 606 from:
data_loader_obj = ADNIDataLoader(...)
# To:
data_loader_obj = ADNIDataLoaderScanLevel(...)