import sys
ROOT_PATH='/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox'
sys.path.append(ROOT_PATH)
from pyexpat import model
import pytest
from sympy import im
import tensorflow as tf
import numpy as np
from csmt.attacks.evasion import MalwareGDTensorFlow
from csmt.estimators.classification.tensorflow import TensorFlowV2Classifier


def fix_get_synthetic_data():
    """
    As real malware is hard to share, generate random data of the correct size
    """
    # generate dummy data
    padding_char = 256
    maxlen = 2 ** 20

    # batch of 5 datapoints
    synthetic_data = np.ones((5, maxlen), dtype=np.uint16) * padding_char

    size_of_original_files = [
        int(maxlen * 0.1),  # 1 sample significantly smaller than the maxlen
        int(maxlen * 1.5),  # 1 sample larger then maxlen
        int(maxlen * 0.95),  # 1 sample close to the maximum of the maxlen
        int(maxlen),  # 1 sample at the maxlen
        int(maxlen),
    ]  # 1 sample at the maxlen, this will be assigned a benign label and
    # should not be perturbed by the attack.

    # two class option, later change to binary when ART is generally updated.
    y = np.zeros((5, 1))
    y[0:4] = 1  # assign the first 4 datapoints to be labeled as malware

    # fill in with random numbers
    for i, size in enumerate(size_of_original_files):
        if size > maxlen:
            size = maxlen
        synthetic_data[i, 0:size] = np.random.randint(low=0, high=256, size=(1, size))

    # set the DOS header values:
    synthetic_data[:, 0:2] = [77, 90]
    synthetic_data[:, int(0x3C) : int(0x40)] = 0  # zero the pointer location
    synthetic_data[:, int(0x3C)] = 44  # put in a dummy pointer
    synthetic_data[:, int(0x3C) + 1] = 1  # put in a dummy pointer
    return synthetic_data, y, np.asarray(size_of_original_files)

def fix_make_dummy_model():
    """
    Create a random model for testing
    """

    def get_prediction_model(param_dic):
        """
        Model going from embeddings to predictions so we can easily optimise the embedding malware embedding.
        Needs to have the same structure as the target model.
        Populated here with "standard" parameters.
        """
        inp = tf.keras.layers.Input(
            shape=(
                param_dic["maxlen"],
                param_dic["embedding_size"],
            )
        )
        filt = tf.keras.layers.Conv1D(
            filters=128,
            kernel_size=500,
            strides=500,
            use_bias=True,
            activation="relu",
            padding="valid",
            name="filt_layer",
        )(inp)
        attn = tf.keras.layers.Conv1D(
            filters=128,
            kernel_size=500,
            strides=500,
            use_bias=True,
            activation="sigmoid",
            padding="valid",
            name="attn_layer",
        )(inp)
        gated = tf.keras.layers.Multiply()([filt, attn])
        feat = tf.keras.layers.GlobalMaxPooling1D()(gated)
        dense = tf.keras.layers.Dense(128, activation="relu", name="dense_layer")(feat)
        output = tf.keras.layers.Dense(1, name="output_layer")(dense)
        return tf.keras.Model(inputs=inp, outputs=output)

    param_dic = {"maxlen": 2 ** 20, "input_dim": 257, "embedding_size": 8}
    prediction_model = get_prediction_model(param_dic)

    model_weights = np.random.normal(loc=0, scale=1.0, size=(257, 8))

    classifier = TensorFlowV2Classifier(
        model=prediction_model,
        nb_classes=2,
        loss_object=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        input_shape=(param_dic["maxlen"], param_dic["embedding_size"]),
    )

    return classifier, model_weights

def test_no_perturbation(fix_get_synthetic_data,fix_make_dummy_model):
    """
    Assert that with 0 perturbation the data is unmodified
    """
    param_dic = {"maxlen": 2 ** 20, "input_dim": 257, "embedding_size": 8}
    # First check: with no perturbation the malware of sufficient size, and benign files, should be unperturbed
    # fix_make_dummy_model_=fix_make_dummy_model
    # fix_get_synthetic_data_=fix_get_synthetic_data
    attack = MalwareGDTensorFlow(
        classifier=fix_make_dummy_model[0], embedding_weights= fix_make_dummy_model[1], l_0=0, param_dic=param_dic
    )

    attack.l_0 = 0
    x = np.copy(fix_get_synthetic_data[0])
    y = np.copy(fix_get_synthetic_data[1])
    size_of_files = np.copy(fix_get_synthetic_data[2])

    adv_x, adv_y, adv_sizes = attack.pull_out_valid_samples(x, y, size_of_files)

    # We should only have 3 files as the following cannot be converted to valid adv samples:
    #   2nd datapoint (file too large to support any modifications)
    #   5th datapoint (benign file)

    assert len(adv_x) == 3

    adv_x = attack.generate(adv_x, adv_y, adv_sizes)
    print(adv_x)

def test_append_attack(fix_get_synthetic_data, fix_make_dummy_model):
    """
    Check append attack wih a given l0 budget
    """

    param_dic = {"maxlen": 2 ** 20, "input_dim": 257, "embedding_size": 8}
    l0_budget = 1250
    attack = MalwareGDTensorFlow(
        classifier=fix_make_dummy_model[0],
        embedding_weights=fix_make_dummy_model[1],
        l_0=l0_budget,
        param_dic=param_dic,
    )

    x = np.copy(fix_get_synthetic_data[0])
    y = np.copy(fix_get_synthetic_data[1])
    size_of_files = np.copy(fix_get_synthetic_data[2])

    adv_x, adv_y, adv_sizes = attack.pull_out_valid_samples(x, y, size_of_files)

    # We should only have 2 files as the following cannot be converted to valid adv samples:
    #   2nd datapoint (file too large to support any modifications)
    #   4th datapoint (file to large to support append attacks)
    #   5th datapoint (benign file)

    assert len(adv_x) == 2

    adv_x = attack.generate(adv_x, adv_y, adv_sizes)
    print(adv_x)

def test_slack_attack(fix_get_synthetic_data, fix_make_dummy_model):
    """
    Testing modification of certain regions in the PE file
    """
    # Third check: Slack insertion attacks.

    def generate_synthetic_slack_regions(size):
        """
        Generate 4 slack regions per sample, each of size 250.
        """

        batch_of_slack_starts = []
        batch_of_slack_sizes = []

        for _ in range(5):
            size_of_slack = []
            start_of_slack = []
            start = 0
            for _ in range(4):
                start += 1000
                start_of_slack.append(start)
                size_of_slack.append(size)
            batch_of_slack_starts.append(start_of_slack)
            batch_of_slack_sizes.append(size_of_slack)
        return batch_of_slack_starts, batch_of_slack_sizes

    param_dic = {"maxlen": 2 ** 20, "input_dim": 257, "embedding_size": 8}
    # First check: with no perturbation the malware of sufficient size, and benign files, should be unperturbed
    attack = MalwareGDTensorFlow(
        classifier=fix_make_dummy_model[0], embedding_weights=fix_make_dummy_model[1], l_0=0, param_dic=param_dic
    )

    l0_budget = 1250
    attack.l_0 = l0_budget
    x = np.copy(fix_get_synthetic_data[0])
    y = np.copy(fix_get_synthetic_data[1])
    size_of_files = np.copy(fix_get_synthetic_data[2])

    batch_of_section_starts, batch_of_section_sizes = generate_synthetic_slack_regions(size=250)
    adv_x, adv_y, adv_sizes, batch_of_section_starts, batch_of_section_sizes = attack.pull_out_valid_samples(
        x,
        y,
        sample_sizes=size_of_files,
        perturb_starts=batch_of_section_starts,
        perturb_sizes=batch_of_section_sizes,
    )

    # We should only have 2 files as the following cannot be converted to valid adv samples:
    #   2nd datapoint (file too large to support any modifications)
    #   4th datapoint (attack requires appending 250 bytes to end of file which this datapoint cannot support)
    #   5th datapoint (benign file)
    assert len(adv_x) == 2

    adv_x = attack.generate(
        adv_x, adv_y, adv_sizes, perturb_sizes=batch_of_section_sizes, perturb_starts=batch_of_section_starts
    )
    print(adv_x)

def test_dos_header_attack(fix_get_synthetic_data, fix_make_dummy_model):
    """
    Test the DOS header attack modifies the correct regions
    """
    # 5th check: DOS header attack
    param_dic = {"maxlen": 2 ** 20, "input_dim": 257, "embedding_size": 8}
    # First check: with no perturbation the malware of sufficient size, and benign files, should be unperturbed
    l0_budget = 290
    attack = MalwareGDTensorFlow(
        classifier=fix_make_dummy_model[0],
        embedding_weights=fix_make_dummy_model[1],
        l_0=l0_budget,
        param_dic=param_dic,
    )
    x = np.copy(fix_get_synthetic_data[0])
    y = np.copy(fix_get_synthetic_data[1])
    size_of_files = np.copy(fix_get_synthetic_data[2])

    dos_starts, dos_sizes = attack.get_dos_locations(x)

    adv_x, adv_y, adv_sizes, batch_of_section_starts, batch_of_section_sizes = attack.pull_out_valid_samples(
        x, y, sample_sizes=size_of_files, perturb_starts=dos_starts, perturb_sizes=dos_sizes
    )

    # should have 3 files. Samples which are excluded are:
    #   2nd datapoint (file to large to support any modifications)
    #   5th datapoint (benign file)

    assert len(adv_x) == 3

    adv_x = attack.generate(
        adv_x, adv_y, adv_sizes, perturb_sizes=batch_of_section_sizes, perturb_starts=batch_of_section_starts
    )

    j = 0
    for i in range(len(fix_get_synthetic_data[2])):
        if i in [0, 2, 3]:
            assert np.array_equal(adv_x[j, 0:2], [77, 90])

            # we should have 58 bytes that were perturbed between the magic number and the pointer
            assert not np.array_equal(adv_x[j, 2 : int(0x3C)], fix_get_synthetic_data[0][i, 2 : int(0x3C)])

            # dummy pointer should be unchanged
            assert np.array_equal(adv_x[j, int(0x3C) : int(0x3C) + 4], [44, 1, 0, 0])

            # the remaining perturbation 290 - 58 = 232 is in the rest of the DOS header
            assert not np.array_equal(
                adv_x[j, int(0x3C) + 4 : int(0x3C) + 4 + 232],
                fix_get_synthetic_data[0][i, int(0x3C) + 4 : int(0x3C) + 4 + 232],
            )

            # rest of the file is unchanged
            assert np.array_equal(
                adv_x[j, int(0x3C) + 4 + 232 :], fix_get_synthetic_data[0][i, int(0x3C) + 4 + 232 :]
            )
            j += 1

if __name__ == '__main__':
    fix_get_synthetic_data=fix_get_synthetic_data()
    # print(fix_get_synthetic_data[0].shape)
    # print(fix_get_synthetic_data[0][0])
    fix_make_dummy_model=fix_make_dummy_model()
    # test_no_perturbation(fix_get_synthetic_data,fix_make_dummy_model)
    test_append_attack(fix_get_synthetic_data,fix_make_dummy_model)

    
    # path='/Users/zhanghangsheng/Documents/my_code/secml_malware-master/secml_malware/data/malware_samples/QvodSetuPuls23.exe'
    # with open(path, "rb") as file_handle:
    #     code = file_handle.read()
    #     np_code=np.frombuffer(code, dtype=np.uint8)
    #     print(np_code)

    

