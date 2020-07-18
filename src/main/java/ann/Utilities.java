package ann;

import org.apache.commons.math3.analysis.UnivariateFunction;
import org.apache.commons.math3.exception.DimensionMismatchException;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import java.util.Random;
import java.util.function.Function;
import java.util.stream.IntStream;

import static model.Constants.SEED;

public class Utilities {

    public static Random rng = new Random(SEED);

    public static RealMatrix generateRandomRealMatrix(int numRows, int numCols) {
        double[][] data = new double[numRows][numCols];
        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[i].length; j++) {
                data[i][j] = rng.nextDouble();
            }
        }
        return new Array2DRowRealMatrix(data);
    }

    public static RealMatrix applyToEachColumn(RealMatrix matrix, UnivariateFunction function) {
        RealMatrix result = new Array2DRowRealMatrix(matrix.getRowDimension(), matrix.getColumnDimension());
        IntStream.range(0, matrix.getColumnDimension()).parallel().forEach(i ->
                result.setColumnVector(i, matrix.getColumnVector(i).map(function))
        );
        return result;
    }

    public static RealMatrix applyToEachRow(RealMatrix matrix, UnivariateFunction function) {
        RealMatrix result = new Array2DRowRealMatrix(matrix.getRowDimension(), matrix.getColumnDimension());
        IntStream.range(0, matrix.getRowDimension()).parallel().forEach(i ->
                result.setRowVector(i, matrix.getRowVector(i).map(function))
        );
        return result;
    }

    public static RealMatrix applyToEachElement(RealMatrix matrix, UnivariateFunction function) {
        RealMatrix result = new Array2DRowRealMatrix(matrix.getRowDimension(), matrix.getColumnDimension());
        IntStream.range(0, matrix.getRowDimension()).parallel().forEach(i ->
                IntStream.range(0, matrix.getColumnDimension()).parallel().forEach(j ->
                        result.setEntry(i, j, function.value(matrix.getEntry(i, j)))
                )
        );
        return result;
    }

    public static RealMatrix ebeMultiply(RealMatrix m1, RealMatrix m2) {
        if (m1.getColumnDimension() != m2.getColumnDimension()){
            throw new DimensionMismatchException(m1.getColumnDimension(), m2.getColumnDimension());
        }
        if (m1.getRowDimension() != m2.getRowDimension()) {
            throw new DimensionMismatchException(m1.getRowDimension(), m2.getRowDimension());
        }
        RealMatrix result = new Array2DRowRealMatrix(m1.getRowDimension(), m1.getColumnDimension());
        IntStream.range(0, m1.getRowDimension()).parallel().forEach(i ->
                IntStream.range(0, m1.getColumnDimension()).parallel().forEach(j ->
                    result.setEntry(i, j, m1.getEntry(i, j) * m2.getEntry(i, j))
                )
        );
        return result;
    }
}
