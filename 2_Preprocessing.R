# load libraries
library(data.table)

# set working directory
setwd('C:\\Users\\Peter A Hall\\Documents\\GitHub\\knn_stock_trading')

# prepare dataframe
dfprep <- read.csv("ohlc_5min_interval.csv")

# drop unneeded columns
ohlc <- dfprep[,1:4]

# transpose dataframe and convert to vector
# structure created is: open1, high1, low1, close1, open2, high2, low2, close2, etc...
ohlcVec <- as.vector(t(ohlc))

# create subsetting function
subsetVector <- function(y, i) {
	last <- i + 43
	return(y[i:last])
}

# create dataframe
ohlcList <- lapply(seq_along(ohlcVec), subsetVector, y = ohlcVec)
ohlcDF <- as.data.frame(t(as.data.frame(ohlcList)))

# clean up dataframe
rownames(ohlcDF) <- NULL
ohlcDF <- na.omit(ohlcDF)

# standardize by row
ohlcStandard <- apply(ohlcDF, 1, scale)

# write training data to csv
write.csv(ohlcStandard,"preparedData.csv", row.names = FALSE)
