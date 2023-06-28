# from tts.data import synthetic_tumor_data

# def test_tumor_data_wilkerson():
#     X, ts, ys = synthetic_tumor_data(100,20,1.0,0.0,0,'wilkerson')
#     assert X.shape == (100,4)
#     assert len(ts) == 100
#     assert len(ys) == 100
#     assert ts[0].shape == (20,)
#     assert ys[0].shape == (20,)

# def test_tumor_data_geng():
#     X, ts, ys = synthetic_tumor_data(100,20,1.0,0.0,0,'geng')
#     assert X.shape == (100,5)
#     assert len(ts) == 100
#     assert len(ys) == 100
#     assert ts[0].shape == (20,)
#     assert ys[0].shape == (20,)