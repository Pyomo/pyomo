Frequently asked questions
==========================

#. Why not use Pandas DataFrames?

Pandas DataFrames are a natural data structure for storing "columns" of
time series data. These columns, or individual time series, could each represent
the data for a single variable. This is very similar to the TimeSeriesData
class introduced in this package.
The reason a new data structure is introduced is primarily that a DataFrame
does not provide any utility for converting labels into a consistent format,
as TimeSeriesData does by accepting variables, strings, slices, etc.
as keys and converting them into the form of a time-indexed ComponentUID.
Also, DataFrames do not have convenient analogs for scalar data and
time interval data, which this package provides as the ScalarData
and IntervalData classes with very similar APIs to TimeSeriesData.
