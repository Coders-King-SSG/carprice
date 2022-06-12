import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
def app(car_df):
	st.header('Visualize Data')
	st.set_option('deprecation.showPyplotGlobalUse', False)
	st.subheader('Scatter Plot')
	ftr = st.multiselect('Select the inputs', ('carwidth', 'enginesize', 'horsepower', 'drivewheel_fwd', 'car_company_buick'))
	for i in ftr:
		st.subheader(f'Scatter plot between {ftr} and Price')
		plt.figure(figsize=(10, 5))
		sns.scatterplot(x=i, y='price', data = car_df, edgecolor='orange')
		st.pyplot()
		st.subheader('Visualization Selector')
		ch = st.multiselect('Select the chart', ('Histogram', 'Boxplot', 'Correlation Heatmap'))
		if 'Histogram' in ch:
			st.subheader('Histogram')
			plt.figure(figsize=(10, 5))
			plt.hist(car_df[i], bins='sturges', edgecolor='#f80')
			st.pyplot()
		if 'Boxplot' in ch:
			st.subheader('Boxplot')
			plt.figure(figsize=(10, 5))
			sns.boxplot(car_df[i], color='#f80')
			st.pyplot()
		if 'Correlation Heatmap' in ch:
			st.subheader('Correlation Heatmap')
			plt.figure(figsize=(10, 5))
			sns.heatmap(car_df.corr(), annot=True)
			st.pyplot()