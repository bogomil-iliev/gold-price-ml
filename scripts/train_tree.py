dt = DecisionTreeRegressor()
dt.fit(X_train, y_train)
yhat = dt.predict(X_test)



model = DecisionTreeRegressor()
regr_trans = TransformedTargetRegressor(regressor=model, transformer=QuantileTransformer(output_distribution='normal'))
regr_trans.fit(X_train, y_train)
yhat = regr_trans.predict(X_test)



print('Mean Absolute Error(DT): ', mean_absolute_error(y_test, yhat))
print('Root Mean Squared Error(DT): ', np.sqrt(mean_squared_error(y_test, yhat)))
acurracyScoreDt=r2_score(y_test, yhat)
acurracyScoreDt="{:.0%}".format(acurracyScoreDt)
print('R2 Score(DT): ' + str(acurracyScoreDt))

plt.title("Distribution of Real vs. Predicted Values (Decision Tree)")
ax1=sns.distplot(y_test, hist=False)
sns.distplot(yhat, hist=False, ax=ax1)
plt.show()

