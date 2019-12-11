import numpy as np

X = np.arange(1, 10, .5)
Y = np.arange(10, 500, 20)
Z = np.array([[0.42, 0.5366666666666666, 0.72, 0.83, 0.8633333333333333, 0.8566666666666667, 0.8366666666666667, 0.8333333333333334, 0.82, 0.81, 0.7833333333333333, 0.77, 0.7633333333333333, 0.7566666666666667, 0.7433333333333333, 0.7166666666666667, 0.7133333333333334, 0.6866666666666666], [0.46, 0.7033333333333334, 0.8266666666666667, 0.88, 0.8633333333333333, 0.85, 0.8133333333333334, 0.7966666666666666, 0.7766666666666666, 0.77, 0.7466666666666667, 0.74, 0.7166666666666667, 0.6966666666666667, 0.6933333333333334, 0.6866666666666666, 0.68, 0.6733333333333333], [0.4533333333333333, 0.6866666666666666, 0.8266666666666667, 0.8866666666666667, 0.9066666666666666, 0.91, 0.91, 0.88, 0.8566666666666667, 0.8266666666666667, 0.79, 0.7633333333333333, 0.7266666666666667, 0.7166666666666667, 0.7166666666666667, 0.7066666666666667, 0.7033333333333334, 0.6866666666666666], [0.51, 0.76, 0.8633333333333333, 0.8966666666666666, 0.9066666666666666, 0.89, 0.8466666666666667, 0.8133333333333334, 0.7933333333333333, 0.7633333333333333, 0.7433333333333333, 0.7166666666666667, 0.7033333333333334, 0.7, 0.6933333333333334, 0.6866666666666666, 0.68, 0.68], [0.5133333333333333, 0.7433333333333333, 0.86, 0.9, 0.9166666666666666, 0.8866666666666667, 0.8433333333333334, 0.8166666666666667, 0.7966666666666666, 0.7666666666666667, 0.76, 0.73, 0.72, 0.7, 0.6966666666666667, 0.6833333333333333, 0.6833333333333333, 0.6733333333333333], [0.49666666666666665, 0.7133333333333334, 0.8333333333333334, 0.89, 0.9066666666666666, 0.9133333333333333, 0.8933333333333333, 0.8533333333333334, 0.8266666666666667, 0.78, 0.76, 0.7366666666666667, 0.7233333333333334, 0.71, 0.6966666666666667, 0.6966666666666667, 0.69, 0.6766666666666666], [0.46, 0.7066666666666667, 0.83, 0.9, 0.9233333333333333, 0.9433333333333334, 0.9333333333333333, 0.9, 0.8833333333333333, 0.8533333333333334, 0.8, 0.7866666666666666, 0.75, 0.7366666666666667, 0.7266666666666667, 0.72, 0.71, 0.7], [0.49, 0.7066666666666667, 0.8366666666666667, 0.8966666666666666, 0.92, 0.93, 0.93, 0.89, 0.86, 0.8166666666666667, 0.79, 0.7666666666666667, 0.74, 0.7266666666666667, 0.71, 0.71, 0.7, 0.69], [0.53, 0.7533333333333333, 0.8666666666666667, 0.9233333333333333, 0.9333333333333333, 0.9333333333333333, 0.8966666666666666, 0.87, 0.83, 0.7833333333333333, 0.7433333333333333, 0.7266666666666667, 0.7266666666666667, 0.71, 0.6966666666666667, 0.6866666666666666, 0.6866666666666666, 0.68], [0.53, 0.7533333333333333, 0.8666666666666667, 0.92, 0.9433333333333334, 0.95, 0.9033333333333333, 0.8533333333333334, 0.8233333333333334, 0.79, 0.75, 0.7333333333333333, 0.7266666666666667, 0.7233333333333334, 0.7066666666666667, 0.6966666666666667, 0.6933333333333334, 0.69], [0.5266666666666666, 0.7533333333333333, 0.86, 0.9066666666666666, 0.93, 0.9133333333333333, 0.8633333333333333, 0.83, 0.8033333333333333, 0.7766666666666666, 0.7566666666666667, 0.7233333333333334, 0.7133333333333334, 0.7, 0.6933333333333334, 0.6866666666666666, 0.6866666666666666, 0.6866666666666666], [0.5133333333333333, 0.75, 0.8566666666666667, 0.9166666666666666, 0.9366666666666666, 0.93, 0.9133333333333333, 0.88, 0.8433333333333334, 0.81, 0.7566666666666667, 0.7433333333333333, 0.7266666666666667, 0.7233333333333334, 0.7133333333333334, 0.7, 0.6966666666666667, 0.69], [0.51, 0.7466666666666667, 0.8566666666666667, 0.92, 0.9466666666666667, 0.9366666666666666, 0.8933333333333333, 0.8566666666666667, 0.8233333333333334, 0.78, 0.76, 0.74, 0.73, 0.7233333333333334, 0.7033333333333334, 0.7033333333333334, 0.6933333333333334, 0.6833333333333333], [0.5566666666666666, 0.7633333333333333, 0.8666666666666667, 0.92, 0.9366666666666666, 0.92, 0.8833333333333333, 0.84, 0.8033333333333333, 0.7633333333333333, 0.7466666666666667, 0.7333333333333333, 0.7133333333333334, 0.7033333333333334, 0.6866666666666666, 0.69, 0.6833333333333333, 0.6833333333333333], [0.5533333333333333, 0.7466666666666667, 0.86, 0.9266666666666666, 0.9366666666666666, 0.9033333333333333, 0.8633333333333333, 0.83, 0.7933333333333333, 0.76, 0.7433333333333333, 0.7266666666666667, 0.71, 0.6933333333333334, 0.69, 0.69, 0.68, 0.68], [0.55, 0.7533333333333333, 0.85, 0.9266666666666666, 0.9233333333333333, 0.9, 0.8633333333333333, 0.8333333333333334, 0.7733333333333333, 0.76, 0.73, 0.73, 0.7033333333333334, 0.6966666666666667, 0.69, 0.6866666666666666, 0.68, 0.68], [0.5533333333333333, 0.76, 0.86, 0.9266666666666666, 0.94, 0.9133333333333333, 0.88, 0.8366666666666667, 0.7966666666666666, 0.7633333333333333, 0.74, 0.73, 0.7166666666666667, 0.7033333333333334, 0.6966666666666667, 0.69, 0.6833333333333333, 0.6766666666666666], [0.5366666666666666, 0.7433333333333333, 0.8566666666666667, 0.92, 0.93, 0.9166666666666666, 0.8933333333333333, 0.85, 0.81, 0.77, 0.75, 0.7433333333333333, 0.73, 0.7066666666666667, 0.6933333333333334, 0.6966666666666667, 0.69, 0.68], [0.5366666666666666, 0.7366666666666667, 0.85, 0.92, 0.9333333333333333, 0.9233333333333333, 0.89, 0.8433333333333334, 0.8066666666666666, 0.76, 0.75, 0.73, 0.72, 0.6966666666666667, 0.7, 0.6833333333333333, 0.68, 0.68], [0.5433333333333333, 0.7433333333333333, 0.8566666666666667, 0.92, 0.9433333333333334, 0.9233333333333333, 0.9, 0.8666666666666667, 0.8166666666666667, 0.7866666666666666, 0.7566666666666667, 0.7433333333333333, 0.7233333333333334, 0.71, 0.6933333333333334, 0.6933333333333334, 0.6833333333333333, 0.6833333333333333], [0.5233333333333333, 0.7566666666666667, 0.8533333333333334, 0.9166666666666666, 0.9366666666666666, 0.9233333333333333, 0.9033333333333333, 0.8533333333333334, 0.8066666666666666, 0.7666666666666667, 0.75, 0.7333333333333333, 0.7233333333333334, 0.7133333333333334, 0.6933333333333334, 0.6933333333333334, 0.6866666666666666, 0.6833333333333333], [0.5333333333333333, 0.7533333333333333, 0.86, 0.92, 0.9433333333333334, 0.9366666666666666, 0.9033333333333333, 0.8633333333333333, 0.81, 0.7766666666666666, 0.7533333333333333, 0.7433333333333333, 0.7266666666666667, 0.7133333333333334, 0.6833333333333333, 0.6866666666666666, 0.6833333333333333, 0.6833333333333333], [0.5433333333333333, 0.7433333333333333, 0.8433333333333334, 0.9233333333333333, 0.9433333333333334, 0.9333333333333333, 0.8933333333333333, 0.86, 0.8133333333333334, 0.7866666666666666, 0.7533333333333333, 0.7433333333333333, 0.7266666666666667, 0.71, 0.6933333333333334, 0.6966666666666667, 0.69, 0.6833333333333333], [0.5366666666666666, 0.7466666666666667, 0.8533333333333334, 0.91, 0.9366666666666666, 0.92, 0.8833333333333333, 0.8433333333333334, 0.8066666666666666, 0.7633333333333333, 0.7533333333333333, 0.7366666666666667, 0.7233333333333334, 0.7, 0.6933333333333334, 0.6933333333333334, 0.68, 0.68], [0.5533333333333333, 0.78, 0.8633333333333333, 0.9233333333333333, 0.9366666666666666, 0.9166666666666666, 0.89, 0.84, 0.8, 0.7766666666666666, 0.7433333333333333, 0.7366666666666667, 0.72, 0.7066666666666667, 0.6933333333333334, 0.6866666666666666, 0.6833333333333333, 0.6866666666666666]])