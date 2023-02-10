package Util;

import java.util.Arrays;
import java.util.stream.Collectors;

public class Shape {
    private final int[] dimensions;
    private int size = 0;

    public Shape(int[] dimensions) {
        this.dimensions = dimensions;

        // Shape contains a dimension of size 0
        boolean containsSizeZeroDim = Arrays
                .stream(this.dimensions).anyMatch((dim) -> dim == 0);

        // Verify that the number and size of dimensions is > 0
        if (getDim() > 0 && !containsSizeZeroDim) {
            size = Arrays.stream(dimensions)
                    .reduce(1, (product, dim) -> product * dim);
        }
    }

    /**
     * Get the dimensionality of the shape.
     * @return The number of dimensions.
     */
    public int getDim() {
        return dimensions.length;
    }

    /**
     * Get the total size of the shape.
     * @return Number of elements in the shape.
     */
    public int getSize() {
        return size;
    }

    /**
     * Create a new shape.
     * @param dimensions list of the sizes of each dimension.
     * @return A new instance of Shape.
     */
    public static Shape build(int... dimensions) {
        return new Shape(dimensions);
    }

    public String toString() {
        String joinedDims = Arrays
                .stream(dimensions)
                .mapToObj(Integer::toString)
                .collect(Collectors.joining(","));

        return String.format("(%s)", joinedDims);
    }
}
