function [zeroMeanOfAllDataSet] = calculateZeroMean(allDataSet)
    zeroMeanOfAllDataSet = allDataSet - mean(allDataSet);
end