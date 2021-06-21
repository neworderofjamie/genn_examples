import mnist_mb
import time
from mnist_mb_para import para


para["NUM_STIM_TRAIN"]= 50000
para["NUM_STIM_TEST"]= 10000
#para["RECORD_PN_SPIKES"]= True
#para["RECORD_KC_SPIKES"]= True
#para["RECORD_GGN_SPIKES"]= True
para["rho"]= 0.01
#para["plot"]= True
for NKC in [20000 ]:
    for nKC in [ 20, 50, 100 ]: 
        for wMax in [ 5, 10, 20 ]:
            for eta in [ 0.00005, 0.0001, 0.0002]:
                para["TRAIN"]= True
                para["RECORD_MBON_SPIKES"]= False
                para["NUM_KC"] = NKC
                para["num_KC"]= nKC
                para["wMax"]= wMax*(1.0/(0.44324564+1.07*nKC))
                para["eta"]= eta*(1.0/(0.44324564+1.07*nKC))/0.04578126
                today= time.strftime("%Y%m%d")
                para["basename"]= today+"-NKC{}-nKC{}-wMax{}-eta{}".format(para["NUM_KC"],para["num_KC"],para["wMax"],para["eta"])
                g= mnist_mb.run_mb(para)
                para["TRAIN"]= False
                para["RECORD_MBON_SPIKES"]= True
                mnist_mb.run_mb(para,g)
