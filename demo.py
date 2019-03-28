from lib.datasets.read_data import data_google
from lib.network.HMM import HMM
from novamind.ops.text_ops import *

# data = data_google()
# data.call_train()

def run_hmm():
    hmm = HMM()
    # msn_data_google = text_read("data/Google/txt/msn.txt")
    # msn_data_google_nns = text_read("data/Google/txt/msn_nns.txt")
    msn_without_lstm = text_read("data/Google/txt/without_lstm.txt")

    # msn_data_google = [eval(data) for data in msn_data_google]
    # msn_data_google_nns = [eval(data) for data in msn_data_google_nns]
    msn_data_without_lstm = [eval(data) for data in msn_without_lstm]

    # hmm(msn_data_google)
    # hmm(msn_data_google_nns)
    hmm(msn_data_without_lstm)

run_hmm()
