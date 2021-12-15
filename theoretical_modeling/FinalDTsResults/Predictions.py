import pandas as pd
Si_url = "https://raw.githubusercontent.com/mesmith75/ic-teach-kstmumu-public/main/predictions/std_predictions_si.json"
Pi_url = "https://raw.githubusercontent.com/mesmith75/ic-teach-kstmumu-public/main/predictions/std_predictions_pi.json"
df_Si = (pd.read_json(Si_url)).T   # df_Pi[bin][parameter, ex: Fl][val/err]
df_Pi = (pd.read_json(Pi_url)).T   # df_Pi[bin][parameter, ex: Fl][val/err]

def predictions():
    print('Getting Predictions')
    fl_pred_Pi = [d.get('val') for d in df_Pi['FL']]
    fl_pred_Pi_err = [d.get('err') for d in df_Pi['FL']]

    fl_pred_Si = [d.get('val') for d in df_Si['FL']]
    fl_pred_Si_err = [d.get('err') for d in df_Si['FL']]

    Afb_pred = [d.get('val') for d in df_Si['AFB']]
    Afb_pred_err = [d.get('err') for d in df_Si['AFB']]

    At_pred = [d.get('val') for d in df_Si['S3']]
    At_pred_err = [d.get('err') for d in df_Si['S3']]

    Aim_pred = [d.get('val') for d in df_Si['S9']]
    Aim_pred_err = [d.get('err') for d in df_Si['S9']]

    df_pred = pd.DataFrame(data={'fl_Pi':fl_pred_Pi, 'fl_Pi_err':fl_pred_Pi_err,
                              'fl_Si': fl_pred_Si, 'fl_Si_err': fl_pred_Si_err,
                              'afb':Afb_pred, 'afb_err': Afb_pred_err,
                              'at':At_pred, 'at_err': At_pred_err,
                              'aim':Aim_pred, 'aim_err': Aim_pred_err})
    return df_pred