import streamlit as st
import pandas as pd

def app(car_df):
	st.header('View Data')
	with st.beta_expander('View Dataset'):
		st.table(car_df)
	st.subheader('Column Description')
	if st.checkbox('Show summary'):
		st.table(car_df.describe())
	b1, b2, b3 = st.beta_columns(3)
	with b1:
		if st.checkbox('Show all column name'):
			st.table(car_df.columns)
	with b2:
		if st.checkbox('Show all column data'):
			st.table(car_df.dtypes)
	with b3:
		if st.checkbox('Show data summary'):
			col = st.selectbox('Select columns', (car_df.columns))
			st.table(car_df[col])
