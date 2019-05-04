import urllib
import os
import shutil
import pandas as pd
import numpy as np
import sys
import zipfile
import distutils
import distutils.util


def save_zip_data(write_path, zip_url):
    # zip_url = "http://files.grouplens.org/datasets/movielens/ml-20m.zip"
    print('reading response')
    with urllib.request.urlopen(zip_url) as response:
        zip_file = response.read()
    print('writing zip file')
    with open(write_path, 'wb') as f:
        f.write(zip_file)

def maybe_download_and_extract_movie_data(data_dir, force_overwrite=False):
    write_path = os.path.join(data_dir, 'ml-20m.zip')
    zip_url = "http://files.grouplens.org/datasets/movielens/ml-20m.zip"
    if not os.path.isfile(write_path):
        os.makedirs(data_dir, exist_ok=True)
        print("Zip not downloaded. Downloading now...")
        save_zip_data(write_path, zip_url)
        print("Zip downloaded")
    else:
        print("Zip already downloaded")

    extract_destination = os.path.join(data_dir, "ml-20m")
    if os.path.isdir(extract_destination):
        if not force_overwrite:
            print("seems extracted datadir already exists, and not forcing overwrite. Exiting.")
            return
        else:
            print("Deleting extracted-lib and recreating...")
            shutil.rmtree(extract_destination)
    print('unzipping data')
    with zipfile.ZipFile(write_path, 'r') as zip_ref:
        zip_ref.extractall(extract_destination)

    current_dir = os.path.join(data_dir, 'ml-20m', 'ml-20m')
    temp_dir = os.path.join(data_dir, 'ml-20m-temp')
    right_dir = os.path.join(data_dir, 'ml-20m')

    print("Moving stuff where it should be")
    shutil.move(current_dir, temp_dir)
    shutil.rmtree(right_dir)
    shutil.move(temp_dir, right_dir)

    print('all extracted... in right place too')


def maybe_download_and_extract_netflix_data(data_dir, force_overwrite=False):
    # NOTE: This doesn't work, because the URL is wrong. Stupid kaggle.
    # NOTE This works now because I'm hosting the dataset. Hope that's not an expensive thing to do.

    write_path = os.path.join(data_dir, 'netflix-prize.zip')
    zip_url = "https://s3-us-west-2.amazonaws.com/cf-datasets/netflix-prize-data.zip"
    if not os.path.isfile(write_path):
        os.makedirs(data_dir, exist_ok=True)
        print("Zip not downloaded. Downloading now...")
        save_zip_data(write_path, zip_url)
        print("Zip downloaded")
    else:
        print("Zip already downloaded")

    extract_destination = os.path.join(data_dir, "netflix-prize")
    if os.path.isdir(extract_destination):
        if not force_overwrite:
            print("seems extracted datadir already exists, and not forcing overwrite. Exiting.")
            return
        else:
            print("Deleting extracted-lib and recreating...")
            shutil.rmtree(extract_destination)
    print('unzipping data')
    with zipfile.ZipFile(write_path, 'r') as zip_ref:
        zip_ref.extractall(extract_destination)
    print('all extracted...')

def maybe_download_and_extract_msd(data_dir, force_overwrite=False):
    write_path = os.path.join(data_dir, 'msd.zip')
    zip_url = "http://labrosa.ee.columbia.edu/millionsong/sites/default/files/challenge/train_triplets.txt.zip"
    if not os.path.isfile(write_path):
        os.makedirs(data_dir, exist_ok=True)
        print("Zip not downloaded. Downloading now...")
        save_zip_data(write_path, zip_url)
        print("Zip downloaded")
    else:
        print("Zip already downloaded")

    extract_destination = os.path.join(data_dir, "msd")
    if os.path.isdir(extract_destination):
        if not force_overwrite:
            print("seems extracted datadir already exists, and not forcing overwrite. Exiting.")
            return
        else:
            print("Deleting extracted-lib and recreating...")
            shutil.rmtree(extract_destination)
    print('unzipping data')
    with zipfile.ZipFile(write_path, 'r') as zip_ref:
        zip_ref.extractall(extract_destination)
    print('all extracted...')

def munge_netflix_data(data_dir, force_overwrite=False):
    """
    Too tired to work. What I want to do is: iterate through. Whenever I find a line that
    says "4:" or something, I know that's the movie-ID. Save it to movie_id variable.
    Then, you can use that as the first part of a line for that user-item pair.
    Also, I need to add a header. Once all that is good, I'll have a similar file
    to ratings.csv for ml-20m.
    """
    # NOTE: Is it going to be a problem that the user-ids have missing items? I don't know...
    import re
    # for file_number in [1,2,3,4]:
    goal_file_path = os.path.join(data_dir, 'netflix-prize', 'ratings.csv')
    if os.path.exists(goal_file_path):
        if not force_overwrite:
            print("Looks like goal file already exists. Not overwriting.")
            return
        else:
            print("Something is there. Deleting it.")
            os.remove(goal_file_path)

    with open(goal_file_path, 'w') as ratings_file:
        # headers will be in different order than ml-20m, but the way its parsed, it doesn't matter at all.
        ratings_file.write("movieId,userId,rating,timestamp\n")

        for file_number in [1,2,3,4]:
            print("Processing file {}".format(file_number))
            read_file_path = os.path.join(data_dir, 'netflix-prize', 'combined_data_{}.txt'.format(file_number))
            with open(read_file_path, 'r') as data_file:

                movie_id = None
                for line in data_file.readlines():
                    # print(line)
                    # continue
                    matches = re.match(r'^(\d+):\n$', line)

                    if matches:
                        movie_id = matches[1] #the first is the whole match, the second is the part in parens.
                    else:
                        if movie_id is None:
                            raise Exception("movie_id shouldn't be none")
                        line_list = line.split(",")
                        assert len(line_list) == 3
                        # reorder so its the same as the other one...
                        new_line = [line_list[0]]
                        line = str(movie_id)+","+line
                        # print('writing line: {}.format(line))
                        ratings_file.write(line)

def munge_msd(data_dir, force_overwrite=False):
    # It's really user, song, play-count. I need to re-order a bit. Also, timestamp gets ignored,
    # so I'll just put a default. Also also, if it's in the file, it should be included.
    goal_file_path = os.path.join(data_dir, 'msd', 'ratings.csv')
    if os.path.exists(goal_file_path):
        if not force_overwrite:
            print("Looks like goal file already exists. Not overwriting.")
            return
        else:
            print("Something is there. Deleting it.")
            os.remove(goal_file_path)

    with open(goal_file_path, 'w') as ratings_file:
        # headers will be in different order than ml-20m, but the way its parsed, it doesn't matter at all.
        ratings_file.write("movieId,userId,rating,timestamp\n")

        read_file_path = os.path.join(data_dir, 'msd', 'train_triplets.txt')
        with open(read_file_path, 'r') as data_file:
            for line in data_file.readlines():
                line_list = line.strip().split('\t')
                assert len(line_list) == 3
                new_line_list = [line_list[1], line_list[0], line_list[2], "N-A"]
                new_line_string = ",".join(new_line_list) + "\n"
                ratings_file.write(new_line_string)





def load_train_data(csv_file):
    tp = pd.read_csv(csv_file)
    n_users = tp['uid'].max() + 1

    rows, cols = tp['uid'], tp['sid']
    data = sparse.csr_matrix((np.ones_like(rows),
                             (rows, cols)), dtype='float64',
                             shape=(n_users, n_items))
    return data


def load_tr_te_data(csv_file_tr, csv_file_te):
    tp_tr = pd.read_csv(csv_file_tr)
    tp_te = pd.read_csv(csv_file_te)

    start_idx = min(tp_tr['uid'].min(), tp_te['uid'].min())
    end_idx = max(tp_tr['uid'].max(), tp_te['uid'].max())

    rows_tr, cols_tr = tp_tr['uid'] - start_idx, tp_tr['sid']
    rows_te, cols_te = tp_te['uid'] - start_idx, tp_te['sid']

    data_tr = sparse.csr_matrix((np.ones_like(rows_tr),
                             (rows_tr, cols_tr)), dtype='float64', shape=(end_idx - start_idx + 1, n_items))
    data_te = sparse.csr_matrix((np.ones_like(rows_te),
                             (rows_te, cols_te)), dtype='float64', shape=(end_idx - start_idx + 1, n_items))
    return data_tr, data_te


def process_unzipped_data(DATA_DIR,
                          force_overwrite=False,
                          n_heldout_users=10000,
                          discard_ratings_below=3.5,
                          min_users_per_item_to_include=0,
                          min_clicks_per_user_to_include=5):

    pro_dir = os.path.join(DATA_DIR, 'pro_sg')
    if os.path.isdir(pro_dir):
        if force_overwrite:
            print("Deleting processed-directory and recreating...")
            shutil.rmtree(pro_dir)
        else:
            print("pro_sg dir already exists. Exiting.")
            return
    raw_data = pd.read_csv(os.path.join(DATA_DIR, 'ratings.csv'), header=0)


    # In[4]:


    # binarize the data (only keep ratings >= 4)
    raw_data = raw_data[raw_data['rating'] > discard_ratings_below]


    # ### Data splitting procedure

    # - Select 10K users as heldout users, 10K users as validation users, and the rest of the users for training
    # - Use all the items from the training users as item set
    # - For each of both validation and test user, subsample 80% as fold-in data and the rest for prediction

    # In[6]:


    def get_count(tp, id):
        playcount_groupbyid = tp[[id]].groupby(id, as_index=False)
        count = playcount_groupbyid.size()
        return count


    # In[7]:


    def filter_triplets(tp, min_uc=5, min_sc=0):
        # Only keep the triplets for items which were clicked on by at least min_sc users.
        if min_sc > 0:
            itemcount = get_count(tp, 'movieId')
            tp = tp[tp['movieId'].isin(itemcount.index[itemcount >= min_sc])]

        # Only keep the triplets for users who clicked on at least min_uc items
        # After doing this, some of the items will have less than min_uc users, but should only be a small proportion
        if min_uc > 0:
            usercount = get_count(tp, 'userId')
            tp = tp[tp['userId'].isin(usercount.index[usercount >= min_uc])]

        # Update both usercount and itemcount after filtering
        usercount, itemcount = get_count(tp, 'userId'), get_count(tp, 'movieId')
        return tp, usercount, itemcount


    # Only keep items that are clicked on by at least 5 users

    # In[8]:


    raw_data, user_activity, item_popularity = \
        filter_triplets(raw_data,
                        min_uc=min_clicks_per_user_to_include,
                        min_sc=min_users_per_item_to_include)


    # In[9]:


    sparsity = 1. * raw_data.shape[0] / (user_activity.shape[0] * item_popularity.shape[0])

    print("After filtering, there are %d watching events from %d users and %d movies (sparsity: %.3f%%)" %
          (raw_data.shape[0], user_activity.shape[0], item_popularity.shape[0], sparsity * 100))


    # In[10]:


    unique_uid = user_activity.index

    np.random.seed(98765)
    idx_perm = np.random.permutation(unique_uid.size)
    unique_uid = unique_uid[idx_perm]


    # In[11]:


    # create train/validation/test users
    n_users = unique_uid.size
    # n_heldout_users = 10000

    tr_users = unique_uid[:(n_users - n_heldout_users * 2)]
    vd_users = unique_uid[(n_users - n_heldout_users * 2): (n_users - n_heldout_users)]
    te_users = unique_uid[(n_users - n_heldout_users):]


    # In[12]:


    train_plays = raw_data.loc[raw_data['userId'].isin(tr_users)]


    # In[13]:


    unique_sid = pd.unique(train_plays['movieId'])


    # In[14]:


    show2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))
    profile2id = dict((pid, i) for (i, pid) in enumerate(unique_uid))


    # In[15]:



    if not os.path.exists(pro_dir):
        os.makedirs(pro_dir)

    with open(os.path.join(pro_dir, 'unique_sid.txt'), 'w') as f:
        for sid in unique_sid:
            f.write('%s\n' % sid)


    # In[16]:


    def split_train_test_proportion(data, test_prop=0.2):
        data_grouped_by_user = data.groupby('userId')
        tr_list, te_list = list(), list()

        np.random.seed(98765)

        for i, (_, group) in enumerate(data_grouped_by_user):
            n_items_u = len(group)

            if n_items_u >= 5:
                idx = np.zeros(n_items_u, dtype='bool')
                idx[np.random.choice(n_items_u, size=int(test_prop * n_items_u), replace=False).astype('int64')] = True

                tr_list.append(group[np.logical_not(idx)])
                te_list.append(group[idx])
            else:
                tr_list.append(group)

            if i % 1000 == 0:
                print("%d users sampled" % i)
                sys.stdout.flush()

        data_tr = pd.concat(tr_list)
        data_te = pd.concat(te_list)

        return data_tr, data_te


    # In[17]:


    vad_plays = raw_data.loc[raw_data['userId'].isin(vd_users)]
    vad_plays = vad_plays.loc[vad_plays['movieId'].isin(unique_sid)]


    # In[18]:


    vad_plays_tr, vad_plays_te = split_train_test_proportion(vad_plays)


    # In[19]:


    test_plays = raw_data.loc[raw_data['userId'].isin(te_users)]
    test_plays = test_plays.loc[test_plays['movieId'].isin(unique_sid)]


    # In[20]:


    test_plays_tr, test_plays_te = split_train_test_proportion(test_plays)


    # ### Save the data into (user_index, item_index) format

    # In[21]:


    def numerize(tp):
        uid = map(lambda x: profile2id[x], tp['userId'])
        sid = map(lambda x: show2id[x], tp['movieId'])
        return pd.DataFrame(data={'uid': list(uid), 'sid': list(sid)}, columns=['uid', 'sid'])


    # In[22]:


    train_data = numerize(train_plays)
    train_data.to_csv(os.path.join(pro_dir, 'train.csv'), index=False)


    # In[23]:


    vad_data_tr = numerize(vad_plays_tr)
    vad_data_tr.to_csv(os.path.join(pro_dir, 'validation_tr.csv'), index=False)


    # In[24]:


    vad_data_te = numerize(vad_plays_te)
    vad_data_te.to_csv(os.path.join(pro_dir, 'validation_te.csv'), index=False)


    # In[25]:


    test_data_tr = numerize(test_plays_tr)
    test_data_tr.to_csv(os.path.join(pro_dir, 'test_tr.csv'), index=False)


    # In[26]:


    test_data_te = numerize(test_plays_te)
    test_data_te.to_csv(os.path.join(pro_dir, 'test_te.csv'), index=False)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument("--use-noise-morpher", help="Whether to use noise-morphing or not. Defaults to True.", type=lambda x:bool(distutils.util.strtobool(x)), default=defaults["use_noise_morpher"])
    parser.add_argument("--force-overwrite", help="Re-download, extract, and parse data", type=lambda x:bool(distutils.util.strtobool(x)), default=False)
    parser.add_argument("--dataset", help="Which dataset do you want?", type=str, default='ml-20m')
    args = parser.parse_args()

    force_overwrite = args.force_overwrite
    dataset = args.dataset
    assert dataset in ['ml-20m', 'netflix-prize', 'msd', 'all']

    if dataset == 'ml-20m' or dataset == 'all':
        print("Doing ml-20m stuff!")
        maybe_download_and_extract_movie_data("./data", force_overwrite=force_overwrite)
        process_unzipped_data('./data/ml-20m', force_overwrite=force_overwrite)

    if dataset == 'netflix-prize' or dataset == 'all':
        print("Doing netflix-prize stuff!")
        maybe_download_and_extract_netflix_data('./data', force_overwrite=force_overwrite)
        munge_netflix_data('./data', force_overwrite=force_overwrite)
        process_unzipped_data('./data/netflix-prize', force_overwrite=force_overwrite, n_heldout_users=40000)

    if dataset == 'msd' or dataset == 'all':
        maybe_download_and_extract_msd('./data', force_overwrite=force_overwrite)
        munge_msd('./data', force_overwrite=force_overwrite)
        process_unzipped_data(
            './data/msd',
            force_overwrite=force_overwrite,
            n_heldout_users=50000,
            discard_ratings_below=0.0,
            min_users_per_item_to_include=200,
            min_clicks_per_user_to_include=20)

    print("All done!")
    exit()
