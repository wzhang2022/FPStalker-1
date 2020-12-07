import random
from fingerprint import Fingerprint
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegressionCV
from algo import generate_new_id, compute_similarity_fingerprint, \
    candidates_have_same_id, generate_replay_sequence

forbidden_changes = [
    Fingerprint.LOCAL_JS,
    Fingerprint.DNT_JS,
    Fingerprint.COOKIES_JS
]

allowed_changes_with_sim = [
    Fingerprint.USER_AGENT_HTTP,
    Fingerprint.VENDOR,
    Fingerprint.RENDERER,
    Fingerprint.PLUGINS_JS,
    Fingerprint.LANGUAGE_HTTP,
    Fingerprint.ACCEPT_HTTP
]

allowed_changes = [
    Fingerprint.RESOLUTION_JS,
    Fingerprint.ENCODING_HTTP,

]

not_to_test = set([Fingerprint.PLATFORM_FLASH,
                   Fingerprint.PLATFORM_INCONSISTENCY,
                   Fingerprint.PLATFORM_JS,
                   Fingerprint.PLUGINS_JS_HASHED,
                   Fingerprint.SESSION_JS,
                   Fingerprint.IE_DATA_JS,
                   Fingerprint.ADDRESS_HTTP,
                   Fingerprint.BROWSER_FAMILY,
                   Fingerprint.COOKIES_JS,
                   Fingerprint.DNT_JS,
                   Fingerprint.END_TIME,
                   Fingerprint.FONTS_FLASH_HASHED,
                   Fingerprint.GLOBAL_BROWSER_VERSION,
                   Fingerprint.LANGUAGE_FLASH,
                   Fingerprint.LANGUAGE_INCONSISTENCY,
                   Fingerprint.LOCAL_JS,
                   Fingerprint.MINOR_BROWSER_VERSION,
                   Fingerprint.MAJOR_BROWSER_VERSION,
                   Fingerprint.NB_FONTS,
                   Fingerprint.NB_PLUGINS,
                   Fingerprint.COUNTER,
                   Fingerprint.OS,
                   Fingerprint.HOST_HTTP,
                   Fingerprint.ACCEPT_HTTP,
                   Fingerprint.CONNECTION_HTTP,
                   Fingerprint.ENCODING_HTTP,
                   Fingerprint.RESOLUTION_FLASH,
                   # Fingerprint.TIMEZONE_JS,
                   Fingerprint.VENDOR,
                   # Fingerprint.RENDERER
                   ])


def embedding_based(fingerprint_unknown, user_id_to_fps, counter_to_fingerprint, model, lambda_threshold):
    att_ml = set(fingerprint_unknown.val_attributes.keys())
    att_ml = sorted([x for x in att_ml if x not in not_to_test])

    ip_allowed = False
    candidates = list()
    exact_matching = list()
    prediction = None
    for user_id in user_id_to_fps:
        for counter_str in user_id_to_fps[user_id]:
            counter_known = int(counter_str.split("_")[0])
            fingerprint_known = counter_to_fingerprint[counter_known]

            # check fingerprint full hash for exact matching
            if fingerprint_known.hash == fingerprint_unknown.hash:
                exact_matching.append((counter_str, None, user_id))
            elif len(exact_matching) < 1 and fingerprint_known.constant_hash == \
                    fingerprint_unknown.constant_hash:
                # we make the comparison only if same os/browser/platform
                if fingerprint_known.val_attributes[Fingerprint.GLOBAL_BROWSER_VERSION] > \
                        fingerprint_unknown.val_attributes[Fingerprint.GLOBAL_BROWSER_VERSION]:
                    continue

                forbidden_change_found = False
                for attribute in forbidden_changes:
                    if fingerprint_known.val_attributes[attribute] != \
                            fingerprint_unknown.val_attributes[attribute]:
                        forbidden_change_found = True
                        break

                if forbidden_change_found:
                    continue

                candidates.append((counter_str, None, user_id))

    if len(exact_matching) > 0:
        if len(exact_matching) == 1 or candidates_have_same_id(exact_matching):
            return exact_matching[0][2]
    elif len(candidates) > 0:
        # in this case we apply ML
        data = []
        attributes = sorted(fingerprint_unknown.val_attributes.keys())
        new_candidates = []
        for elt in candidates:
            counter = int(elt[0].split("_")[0])
            fingerprint_known = counter_to_fingerprint[counter]
            x_row, _ = compute_similarity_fingerprint(fingerprint_unknown,
                                                      fingerprint_known,
                                                      att_ml, train_mode=False)
            if x_row is not None:
                data.append(x_row)
                new_candidates.append(elt)

        if len(new_candidates) > 0:
            predictions_model = model.predict_proba(data)
            predictions_model = 1.0 - predictions_model
            nearest = (-predictions_model[:, 0]).argsort()[:3]

            max_nearest = 1
            second_proba = None
            for i in range(1, len(nearest)):
                if predictions_model[nearest[i], 0] != predictions_model[nearest[0], 0]:
                    max_nearest = i
                    second_proba = predictions_model[nearest[i], 0]
                    break
            nearest = nearest[:max_nearest]

            diff_enough = True
            if second_proba is not None and predictions_model[nearest[0], 0] < second_proba + 0.1: # 0.1 = diff parameter
                diff_enough = False

            if diff_enough and predictions_model[nearest[0], 0] > lambda_threshold and candidates_have_same_id(
                    [candidates[x] for x in nearest]):
                prediction = new_candidates[nearest[0]][2]

    if prediction is None:
        prediction = generate_new_id()

    return prediction


def train_siamese(fingerprint_dataset, train_data, load=True, model_path="./data/embedding_model"):
    if load:
        model = joblib.load(model_path)
    else:
        counter_to_fingerprint = dict()
        index_to_user_id = dict()
        user_ids = set()
        index = 0

        att_ml = set(fingerprint_dataset[0].val_attributes.keys())
        att_ml = sorted([x for x in att_ml if x not in not_to_test])
        print("att ml:")
        print(att_ml)
        for fingerprint in fingerprint_dataset:
            counter_to_fingerprint[fingerprint.getCounter()] = fingerprint
            if fingerprint.getId() not in user_ids:
                user_ids.add(fingerprint.getId())
                index_to_user_id[index] = fingerprint.getId()
                index += 1

        # just to simplify negative comparisons later
        # we generate multiple replay sequences on train data with different visit frequencies
        # to generate more diverse training data
        print("Start generating training data")
        for visit_frequency in range(1, 10):
            print(visit_frequency)
            train_replay_sequence = generate_replay_sequence(train_data, visit_frequency)
            # we group fingerprints by user id
            user_id_to_fps = dict()
            for elt in train_replay_sequence:
                counter = int(elt[0].split("_")[0])
                fingerprint = counter_to_fingerprint[counter]
                if fingerprint.getId() not in user_id_to_fps:
                    user_id_to_fps[fingerprint.getId()] = []
                user_id_to_fps[fingerprint.getId()].append(fingerprint)

            # we generate the training data
            X, y = [], []
            print("Number of user id: {:d}".format(len(user_id_to_fps)))
            for user_id in user_id_to_fps:
                previous_fingerprint = None
                print("Number of fingerprints: {:d}".format(len(user_id_to_fps[user_id])))
                for fingerprint in user_id_to_fps[user_id]:
                    if previous_fingerprint is not None:
                        x_row, y_row = compute_similarity_fingerprint(fingerprint, previous_fingerprint, att_ml,
                                                                      train_mode=True)
                        print(att_ml)
                        print("Length vector: {:d}".format(len(x_row)))
                        X.append(x_row)
                        y.append(y_row)
                    previous_fingerprint = fingerprint

            # we compute negative rows
            for user_id in user_id_to_fps:
                for fp1 in user_id_to_fps[user_id]:
                    try:
                        compare_with_id = index_to_user_id[random.randint(0, len(index_to_user_id)-1)]
                        compare_with_fp = random.randint(0, len(user_id_to_fps[compare_with_id])-1)
                        fp2 = user_id_to_fps[compare_with_id][compare_with_fp]
                        x_row, y_row = compute_similarity_fingerprint(fp1, fp2, att_ml, train_mode=True)
                        X.append(x_row)
                        y.append(y_row)
                    except Exception as e:
                        pass

        print("Start training model")
        # CHANGE MODEL HERE
        # model = RandomForestClassifier(n_estimators=10, max_features=3, n_jobs=4)
        model = LogisticRegressionCV()

        print("Training data: %d" % len(X))
        model.fit(X, y)
        print("Model trained")
        joblib.dump(model, model_path)
        print("model saved at: %s" % model_path)

    return model