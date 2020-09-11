import os
import csv
import logging
from speechbrain.utils.data_utils import get_all_files

from speechbrain.data_io.data_io import (
    read_wav_soundfile,
    load_pkl,
    save_pkl,
    read_kaldi_lab,
)

logger = logging.getLogger(__name__)
OPT_FILE = "opt_timit_prepare.pkl"
TRAIN_CSV = "train.csv"
DEV_CSV = "dev.csv"
TEST_CSV = "test.csv"
VALID_CSV = "valid.csv"
SAMPLERATE = 16000


# def prepare_timit(
#     data_folder,
#     splits,
#     save_folder,
#     kaldi_ali_tr=None,
#     kaldi_ali_dev=None,
#     kaldi_ali_test=None,
#     kaldi_lab_opts=None,
#     phn_set="39",
#     uppercase=True,
# ):
#     """
#     repares the csv files for the TIMIT dataset.
#
#     Arguments
#     ---------
#     data_folder : str
#         Path to the folder where the original TIMIT dataset is stored.
#     splits : list
#         List of splits to prepare from ['train', 'dev', 'test']
#     save_folder : str
#         The directory where to store the csv files.
#     kaldi_ali_tr : dict, optional
#         Default: 'None'
#         When set, this is the directiory where the kaldi
#         training alignments are stored.  They will be automatically converted
#         into pkl for an easier use within speechbrain.
#     kaldi_ali_dev : str, optional
#         Default: 'None'
#         When set, this is the path to directory where the
#         kaldi dev alignments are stored.
#     kaldi_ali_te : str, optional
#         Default: 'None'
#         When set, this is the path to the directory where the
#         kaldi test alignments are stored.
#     phn_set : {60, 48, 39}, optional,
#         Default: 39
#         The phoneme set to use in the phn label.
#         It could be composed of 60, 48, or 39 phonemes.
#     uppercase : bool, optional
#         Default: False
#         This option must be True when the TIMIT dataset
#         is in the upper-case version.
#
#     Example
#     -------
#     >>> from recipes.TIMIT.timit_prepare import prepare_timit
#     >>> data_folder = 'datasets/TIMIT'
#     >>> splits = ['train', 'dev', 'test']
#     >>> save_folder = 'TIMIT_prepared'
#     >>> prepare_timit(data_folder, splits, save_folder)
#     """
#     conf = {
#         "data_folder": data_folder,
#         "splits": splits,
#         "kaldi_ali_tr": kaldi_ali_tr,
#         "kaldi_ali_dev": kaldi_ali_dev,
#         "kaldi_ali_test": kaldi_ali_test,
#         "save_folder": save_folder,
#         "phn_set": phn_set,
#         "uppercase": uppercase,
#     }
#
#     # Getting speaker dictionary
#     # dev_spk, test_spk = _get_speaker()
#
#     # Avoid calibration sentences
#     avoid_sentences = ["sa1", "sa2"]
#
#     # Setting file extension.
#     extension = [".wav"]
#
#     # Checking TIMIT_uppercase
#     if uppercase:
#         avoid_sentences = [item.upper() for item in avoid_sentences]
#         extension = [item.upper() for item in extension]
#         # dev_spk = [item.upper() for item in dev_spk]
#         # test_spk = [item.upper() for item in test_spk]
#
#     # Setting the save folder
#     if not os.path.exists(save_folder):
#         os.makedirs(save_folder)
#
#     # Setting ouput files
#     save_opt = os.path.join(save_folder, OPT_FILE)
#     save_csv_train = os.path.join(save_folder, TRAIN_CSV)
#     save_csv_dev = os.path.join(save_folder, DEV_CSV)
#     save_csv_test = os.path.join(save_folder, TEST_CSV)
#
#     # Check if this phase is already done (if so, skip it)
#     if skip(splits, save_folder, conf):
#         logger.debug("Skipping preparation, completed in previous run.")
#         return
#
#     # Additional checks to make sure the data folder contains TIMIT
#     _check_timit_folders(uppercase, data_folder)
#
#     msg = "\tCreating csv file for the TIMIT Dataset.."
#     logger.debug(msg)
#
#     # Creating csv file for training data
#     if "train" in splits:
#
#         # Checking TIMIT_uppercase
#         if uppercase:
#             match_lst = extension + ["TRAIN"]
#         else:
#             match_lst = extension + ["train"]
#
#         wav_lst_train = get_all_files(
#             data_folder, match_and=match_lst, exclude_or=avoid_sentences,
#         )
#
#         create_csv(
#             wav_lst_train,
#             save_csv_train,
#             uppercase,
#             data_folder,
#             phn_set,
#             kaldi_lab=kaldi_ali_tr,
#             kaldi_lab_opts=kaldi_lab_opts,
#         )
#
#     # Creating csv file for dev data
#     if "dev" in splits:
#
#         # Checking TIMIT_uppercase
#         if uppercase:
#             match_lst = extension + ["TEST"]
#         else:
#             match_lst = extension + ["test"]
#
#         wav_lst_dev = get_all_files(
#             data_folder,
#             match_and=match_lst,
#             match_or=dev_spk,
#             exclude_or=avoid_sentences,
#         )
#
#         create_csv(
#             wav_lst_dev,
#             save_csv_dev,
#             uppercase,
#             data_folder,
#             phn_set,
#             kaldi_lab=kaldi_ali_dev,
#             kaldi_lab_opts=kaldi_lab_opts,
#         )
#
#     # Creating csv file for test data
#     if "test" in splits:
#
#         # Checking TIMIT_uppercase
#         if uppercase:
#             match_lst = extension + ["TEST"]
#         else:
#             match_lst = extension + ["test"]
#
#         wav_lst_test = get_all_files(
#             data_folder,
#             match_and=match_lst,
#             match_or=test_spk,
#             exclude_or=avoid_sentences,
#         )
#
#         create_csv(
#             wav_lst_test,
#             save_csv_test,
#             uppercase,
#             data_folder,
#             phn_set,
#             kaldi_lab=kaldi_ali_test,
#             kaldi_lab_opts=kaldi_lab_opts,
#         )
#
#     # saving options
#     save_pkl(conf, save_opt)


def prepare_timit(data_folder, save_folder, valid_speaker_count=2):
    """
    Prepares the csv files for the Voicebank dataset.

    Expects the data folder to be the same format as the output of
    ``download_vctk()`` below.

    Arguments
    ---------
    data_folder : str
        Path to the folder where the original Voicebank dataset is stored.
    save_folder : str
        The directory where to store the csv files.
    valid_speaker_count : int
        The number of validation speakers to use (out of 28 in train set).

    Example
    -------
    >>> data_folder = '/path/to/datasets/Voicebank'
    >>> save_folder = 'exp/Voicebank_exp'
    >>> prepare_voicebank(data_folder, save_folder)
    """

    # Setting ouput files
    save_csv_train = os.path.join(save_folder, TRAIN_CSV)
    save_csv_test = os.path.join(save_folder, TEST_CSV)
    # save_csv_valid = os.path.join(save_folder, VALID_CSV)

    # Check if this phase is already done (if so, skip it)
    if skip(save_csv_train, save_csv_test):
        logger.debug("Preparation completed in previous run, skipping.")
        return

    train_clean_folder = os.path.join(data_folder, "TRAIN", "clean")
    train_noisy_folder = os.path.join(data_folder, "TRAIN", "noisy")
    # train_txts = os.path.join(data_folder, "trainset_28spk_txt")
    test_clean_folder = os.path.join(data_folder, "TEST", "clean")
    test_noisy_folder = os.path.join(data_folder, "TEST", "noisy")
    # test_txts = os.path.join(data_folder, "testset_txt")

    # Setting the save folder
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Additional checks to make sure the data folder contains Voicebank
    check_timit_folders(
        train_clean_folder,
        train_noisy_folder,
        # train_txts,
        test_clean_folder,
        test_noisy_folder,
        # test_txts,
    )

    logger.debug("Creating csv files for noisy TIMIT...")

    # Creating csv file for training data
    extension = [".WAV"]
    # valid_speakers = TRAIN_SPEAKERS[:valid_speaker_count]
    wav_lst_train = get_all_files(train_noisy_folder, match_and=extension)
    create_csv(wav_lst_train, save_csv_train, train_noisy_folder, train_clean_folder)

    # Creating csv file for validation data
    # wav_lst_valid = get_all_files(
    #     train_noisy_folder, match_and=extension
    # )
    # create_csv(wav_lst_valid, train_clean_folder)

    # Creating csv file for testing data
    wav_lst_test = get_all_files(test_noisy_folder, match_and=extension)
    create_csv(wav_lst_test, save_csv_test, test_noisy_folder, test_clean_folder)

# def skip(splits, save_folder, conf):
#     """
#     Detects if the timit data_preparation has been already done.
#     If the preparation has been done, we can skip it.
#
#     Returns
#     -------
#     bool
#         if True, the preparation phase can be skipped.
#         if False, it must be done.
#     """
#     # Checking csv files
#     skip = True
#
#     split_files = {
#         "train": TRAIN_CSV,
#         "dev": DEV_CSV,
#         "test": TEST_CSV,
#     }
#     for split in splits:
#         if not os.path.isfile(os.path.join(save_folder, split_files[split])):
#             skip = False
#
#     #  Checking saved options
#     save_opt = os.path.join(save_folder, OPT_FILE)
#     if skip is True:
#         if os.path.isfile(save_opt):
#             opts_old = load_pkl(save_opt)
#             if opts_old == conf:
#                 skip = True
#             else:
#                 skip = False
#         else:
#             skip = False
#
#     return skip

def skip(*filenames):
    """
    Detects if the Voicebank data_preparation has been already done.
    If the preparation has been done, we can skip it.

    Returns
    -------
    bool
        if True, the preparation phase can be skipped.
        if False, it must be done.
    """
    for filename in filenames:
        if not os.path.isfile(filename):
            return False
    return True


# def create_csv(
#     wav_lst,
#     csv_file,
#     uppercase,
#     data_folder,
#     phn_set,
#     kaldi_lab=None,
#     kaldi_lab_opts=None,
#     kaldi_lab_dir=None,
# ):
#     """
#     Creates the csv file given a list of wav files.
#
#     Arguments
#     ---------
#     wav_lst : list
#         The list of wav files of a given data split.
#     csv_file : str
#         The path of the output csv file
#     uppercase : bool
#         Whether this is the uppercase version of timit.
#     data_folder : str
#         The location of the data.
#     kaldi_lab : str, optional
#         Default: None
#         The path of the kaldi labels (optional).
#     kaldi_lab_opts : str, optional
#         Default: None
#         A string containing the options used to compute the labels.
#
#     Returns
#     -------
#     None
#     """
#
#     # Adding some Prints
#     msg = '\t"Creating csv lists in  %s..."' % (csv_file)
#     logger.debug(msg)
#
#     # Reading kaldi labels if needed:
#     snt_no_lab = 0
#     missing_lab = False
#
#     if kaldi_lab is not None:
#
#         lab = read_kaldi_lab(kaldi_lab, kaldi_lab_opts,)
#
#         if not os.path.exists(kaldi_lab_dir):
#             os.makedirs(kaldi_lab_dir)
#
#     csv_lines = [
#         [
#             "ID",
#             "duration",
#             "wav",
#             "wav_format",
#             "wav_opts",
#             "spk_id",
#             "spk_id_format",
#             "spk_id_opts",
#             "phn",
#             "phn_format",
#             "phn_opts",
#             "wrd",
#             "wrd_format",
#             "wrd_opts",
#             "ground_truth_phn_ends",
#             "ground_truth_phn_ends_format",
#             "ground_truth_phn_ends_opts",
#         ]
#     ]
#
#     if kaldi_lab is not None:
#         csv_lines[0].append("kaldi_lab")
#         csv_lines[0].append("kaldi_lab_format")
#         csv_lines[0].append("kaldi_lab_opts")
#
#     # Processing all the wav files in the list
#     for wav_file in wav_lst:
#
#         # Getting sentence and speaker ids
#         spk_id = wav_file.split("/")[-2]
#         snt_id = wav_file.split("/")[-1].replace(".wav", "")
#         snt_id = spk_id + "_" + snt_id
#
#         if kaldi_lab is not None:
#             if snt_id not in lab.keys():
#                 missing_lab = False
#                 msg = (
#                     "\tThe sentence %s does not have a corresponding "
#                     "kaldi label" % (snt_id)
#                 )
#
#                 logger.debug(msg)
#                 snt_no_lab = snt_no_lab + 1
#             else:
#                 snt_lab_path = os.path.join(kaldi_lab_dir, snt_id + ".pkl")
#                 save_pkl(lab[snt_id], snt_lab_path)
#
#             # If too many kaldi labels are missing rise an error
#             if snt_no_lab / len(wav_lst) > 0.05:
#                 err_msg = (
#                     "Too many sentences do not have the "
#                     "corresponding kaldi label. Please check data and "
#                     "kaldi labels (check %s and %s)." % (data_folder, kaldi_lab)
#                 )
#                 logger.error(err_msg, exc_info=True)
#
#         if missing_lab:
#             continue
#
#         # Reading the signal (to retrieve duration in seconds)
#         signal = read_wav_soundfile(wav_file)
#         duration = signal.shape[0] / SAMPLERATE
#
#         # Retrieving words and check for uppercase
#         if uppercase:
#             wrd_file = wav_file.replace(".WAV", ".WRD")
#         else:
#             wrd_file = wav_file.replace(".wav", ".wrd")
#         if not os.path.exists(os.path.dirname(wrd_file)):
#             err_msg = "the wrd file %s does not exists!" % (wrd_file)
#             raise FileNotFoundError(err_msg)
#
#         words = [line.rstrip("\n").split(" ")[2] for line in open(wrd_file)]
#         words = " ".join(words)
#
#         # Retrieving phonemes
#         if uppercase:
#             phn_file = wav_file.replace(".WAV", ".PHN")
#         else:
#             phn_file = wav_file.replace(".wav", ".phn")
#
#         if not os.path.exists(os.path.dirname(phn_file)):
#             err_msg = "the wrd file %s does not exists!" % (phn_file)
#             raise FileNotFoundError(err_msg)
#
#         # Getting the phoneme and ground truth ends lists from the phn files
#         phonemes, ends = get_phoneme_lists(phn_file, phn_set)
#
#         # Composition of the csv_line
#         csv_line = [
#             snt_id,
#             str(duration),
#             wav_file,
#             "wav",
#             "",
#             spk_id,
#             "string",
#             "",
#             str(phonemes),
#             "string",
#             "",
#             str(words),
#             "string",
#             "label:False",
#             str(ends),
#             "string",
#             "label:False",
#         ]
#
#         if kaldi_lab is not None:
#             csv_line.append(snt_lab_path)
#             csv_line.append("pkl")
#             csv_line.append("")
#
#         # Adding this line to the csv_lines list
#         csv_lines.append(csv_line)
#
#     # Writing the csv lines
#     _write_csv(csv_lines, csv_file)
#     msg = "\t%s sucessfully created!" % (csv_file)
#     logger.debug(msg)


def create_csv(wav_lst, csv_file, noisy_folder, clean_folder):
    """
    Creates the csv file given a list of wav files.

    Arguments
    ---------
    wav_lst : list
        The list of wav files.
    csv_file : str
        The path of the output csv file
    clean_folder : str
        The location of parallel clean samples.
    txt_folder : str
        The location of the transcript files.
    """
    logger.debug(f"Creating csv lists in {csv_file}")

    csv_lines = [["ID", "duration"]]
    csv_lines[0].extend(["noisy_wav", "noisy_wav_format", "noisy_wav_opts"])
    csv_lines[0].extend(["clean_wav", "clean_wav_format", "clean_wav_opts"])
    # csv_lines[0].extend(["char", "char_format", "char_opts"])

    # Processing all the wav files in the list
    for wav_file in wav_lst:  # ex:p203_122.wav

        # Example wav_file: p232_001.wav
        snt_id = os.path.basename(wav_file)
        relative_path = os.path.relpath(wav_file, noisy_folder)
        clean_wav = os.path.join(clean_folder, relative_path)

        # Reading the signal (to retrieve duration in seconds)
        signal = read_wav_soundfile(wav_file)
        duration = signal.shape[0] / SAMPLERATE

        # Reading the transcript
        # with open(os.path.join(txt_folder, snt_id + ".txt")) as f:
        #     words = f.read()

        # Strip punctuation and add spaces (excluding repeats).
        # words = words.translate(str.maketrans("", "", string.punctuation))
        # chars = " ".join(words.strip().upper())
        # chars = chars.replace("   ", " <SP> ")
        # chars = re.sub(r"\s{2,}", r" ", chars)
        # chars = re.sub(r"(.) \1", r"\1\1", chars)

        # Composition of the csv_line
        csv_line = [snt_id, str(duration)]
        csv_line.extend([wav_file, "wav", ""])
        csv_line.extend([clean_wav, "wav", ""])
        # csv_line.extend([chars, "string", ""])

        # Adding this line to the csv_lines list
        csv_lines.append(csv_line)

    # Writing the csv lines
    with open(csv_file, mode="w") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )

        for line in csv_lines:
            csv_writer.writerow(line)

    logger.debug(f"{csv_file} successfully created!")


# def _check_timit_folders(uppercase, data_folder):
#     """
#     Check if the data folder actually contains the TIMIT dataset.
#
#     If not, raises an error.
#
#     Returns
#     -------
#     None
#
#     Raises
#     ------
#     FileNotFoundError
#         If data folder doesn't contain TIMIT dataset.
#     """
#
#     # Creating checking string wrt to lower or uppercase
#     if uppercase:
#         test_str = "/TEST/DR1"
#         train_str = "/TRAIN/DR1"
#     else:
#         test_str = "/test/dr1"
#         train_str = "/train/dr1"
#
#     # Checking test/dr1
#     if not os.path.exists(data_folder + test_str):
#         err_msg = (
#             "the folder %s does not exist (it is expected in "
#             "the TIMIT dataset)" % (data_folder + test_str)
#         )
#         raise FileNotFoundError(err_msg)
#
#     # Checking train/dr1
#     if not os.path.exists(data_folder + train_str):
#
#         err_msg = (
#             "the folder %s does not exist (it is expected in "
#             "the TIMIT dataset)" % (data_folder + train_str)
#         )
#         raise FileNotFoundError(err_msg)

def check_timit_folders(*folders):
    """Raises FileNotFoundError if any passed folder does not exist."""
    for folder in folders:
        if not os.path.exists(folder):
            raise FileNotFoundError(
                f"the folder {folder} does not exist (it is expected in "
                "the Voicebank dataset)"
            )
