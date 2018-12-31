import osml.data as osmld
import osml.model as osmlm
import datetime as dt


def predict_rating(data_set_path, model_path, imdb_rating, runtime, num_of_votes, release_date, title_type, genres,
                   directors):
    data_set = osmld.IMDBDataSet(data_set_path, test_share=0)
    model = osmlm.load(model_path)
    passengers_df = data_set.build_observation(imdb_rating, runtime, num_of_votes, release_date, title_type, genres,
                                               directors)
    return model.predict(passengers_df)


def predict_our_ratings(imdb_rating, runtime, num_of_votes, release_date, title_type, genres, directors):
    print('Nina\'s rating:', predict_rating('../data/imdb/ninas_ratings.csv', 'nina_rf.bin', imdb_rating, runtime,
                                            num_of_votes, release_date, title_type, genres, directors)[0])
    print('Viktor\'s rating:', predict_rating('../data/imdb/ratings.csv', 'viktor_rf.bin', imdb_rating, runtime,
                                              num_of_votes, release_date, title_type, genres, directors)[0])


predict_our_ratings(1.6, 105., 74922., dt.datetime(2011, 2, 11), 'movie', 'Documentary, Music',
                    'Jon M. Chu')
