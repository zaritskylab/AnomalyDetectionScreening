

def check_diff_cols(df1, df2):

  in_a = np.setdiff1d(df1.columns, df2.columns)
  in_b = np.setdiff1d(df2.columns, df1.columns)

  diff_cols = np.unique(np.concatenate((in_a, in_b), 0))

  return diff_cols