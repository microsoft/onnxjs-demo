import {Tensor} from 'onnxjs';
import * as tensorTransformUtils from './yoloPostprocess';

// Scalar Test
const actual0 = tensorTransformUtils.scalar(3.14, 'float32');
const data = new Float32Array(1);
data[0] = 3.14;
const expected0 = new Tensor(data,'float32', [1]);
if(!assertTensorEquality(actual0, expected0)) {
    throw new Error('Scalar Test failed');
}

// Zeros Test
const actual1 = tensorTransformUtils.zeros([2,2], 'int32');
const expected1 = new Tensor(new Int32Array(4),'int32', [2,2]);
if(!assertTensorEquality(actual1, expected1)) {
    throw new Error('Zeros Test failed');
}

// Linspace Test
const actual2 = tensorTransformUtils.linspace(0, 9, 10);
const expected2 = new Tensor(Float32Array.from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),'float32', [10]);
if(!assertTensorEquality(actual2, expected2)) {
    throw new Error('Linspace Test failed');
}

// Range Test
const actual3 = tensorTransformUtils.range(0, 9, 2);
const expected3 = new Tensor(Float32Array.from([0, 2, 4, 6, 8]),'float32', [5]);
if(!assertTensorEquality(actual3, expected3)) {
    throw new Error('Range Test failed');
}

// Sigmoid Test
const actual4 = tensorTransformUtils.sigmoid(new Tensor([0, -1, 2, -3], 'float32'));
const expected4 = new Tensor(Float32Array.from([0.5, 0.2689414, 0.8807971, 0.0474259]),'float32', [4]);
if(!assertTensorEquality(actual4, expected4)) {
    throw new Error('Sigmoid Test failed');
}

// Exp Test
const actual5 = tensorTransformUtils.exp(new Tensor([1, 2, -3], 'float32'));
const expected5 = new Tensor(Float32Array.from([2.7182817, 7.3890562, 0.0497871]),'float32', [3]);
if(!assertTensorEquality(actual5, expected5)) {
    throw new Error('Exp Test failed');
}

// Add Test
const actual6 = tensorTransformUtils.add(new Tensor([1, 2, 3], 'float32'), new Tensor([10, 20, 30], 'float32') );
const expected6 = new Tensor(Float32Array.from([11, 22, 33]),'float32', [3]);
if(!assertTensorEquality(actual6, expected6)) {
    throw new Error('Add Test failed');
}

// Sub Test
const actual7 = tensorTransformUtils.sub(new Tensor([1, 2, 3], 'float32'), new Tensor([10, 20, 30], 'float32') );
const expected7 = new Tensor(Float32Array.from([-9, -18, -27]),'float32', [3]);
if(!assertTensorEquality(actual7, expected7)) {
    throw new Error('Sub Test failed');
}

// Mul Test
const actual8 = tensorTransformUtils.mul(new Tensor([1, 2, 3], 'float32'), new Tensor([10, 20, 30], 'float32') );
const expected8 = new Tensor(Float32Array.from([10, 40, 90]),'float32', [3]);
if(!assertTensorEquality(actual8, expected8)) {
    throw new Error('Mul Test failed');
}

// Div Test
const actual9 = tensorTransformUtils.div(new Tensor([10, 20, 30], 'float32'), new Tensor([1, 2, 3], 'float32') );
const expected9 = new Tensor(Float32Array.from([10, 10, 10]),'float32', [3]);
if(!assertTensorEquality(actual9, expected9)) {
    throw new Error('Div Test failed');
}

// Softmax Test
const actual10 = tensorTransformUtils.softmax(new Tensor([2, 4, 6, 1, 2, 3], 'float32', [2, 3]));
const expected10 = new Tensor(Float32Array.from([0.0158762, 0.1173104, 0.8668135, 
                                                 0.0900306, 0.2447284, 0.6652408]),
                                                 'float32', [2, 3]);
if(!assertTensorEquality(actual10, expected10)) {
    throw new Error('Softmax Test failed');
}

// Concat Test
const a1 = new Tensor([1,2], 'float32');
const b1 = new Tensor([3,4], 'float32');
const actual11 = tensorTransformUtils.concat([a1,b1]);
const expected11 = new Tensor(Float32Array.from([1, 2, 3, 4]), 'float32', [4]);
if(!assertTensorEquality(actual11, expected11)) {
    throw new Error('Concat Test failed');
}

// Stack Test
const a2 = new Tensor([1,2], 'float32');
const b2 = new Tensor([3,4], 'float32');
const c2 = new Tensor([5,6], 'float32');
const actual12 = tensorTransformUtils.stack([a2,b2,c2]);
const expected12 = new Tensor(Float32Array.from([1, 2, 3, 4, 5, 6]), 'float32', [3,2]);
if(!assertTensorEquality(actual12, expected12)) {
    throw new Error('Stack Test failed');
}

// Gather Test
const indices = new Tensor(Int32Array.from([1, 3, 3]), 'int32');
const actual13 = tensorTransformUtils.gather(new Tensor(Int32Array.from([1, 2, 3, 4]), 'int32'), indices);
const expected13 = new Tensor(Int32Array.from([2, 4, 4]), 'int32');
if(!assertTensorEquality(actual13, expected13)) {
    throw new Error('Gather Test failed');
}

// Slice Test
const actual13b = tensorTransformUtils.slice(new Tensor(Int32Array.from([1, 2, 3, 4]), 'int32', [2,2]), [1, 0], [1, 2]);
const expected13b = new Tensor(Int32Array.from([3, 4]), 'int32', [1,2]);
if(!assertTensorEquality(actual13b, expected13b)) {
    throw new Error('Slice Test failed');
}

// Tile Test
const actual14 = tensorTransformUtils.tile(new Tensor(Int32Array.from([1, 2, 3, 4]), 'int32', [2,2]), [1,2]);
const expected14 = new Tensor(Int32Array.from([1, 2, 1, 2, 3, 4, 3, 4]), 'int32', [2,4]);
if(!assertTensorEquality(actual14, expected14)) {
    throw new Error('Tile Test failed');
}

// Transpose Test
const actual15 = tensorTransformUtils.transpose(new Tensor([1, 2, 3, 4, 5, 6], 'float32', [2,3]));
const expected15 = new Tensor(Float32Array.from([1, 4, 2, 5, 3, 6]), 'float32', [3,2]);
if(!assertTensorEquality(actual15, expected15)) {
    throw new Error('Transpose Test failed');
}

// ExpandDims Test
const actual16 = tensorTransformUtils.expandDims(new Tensor([1, 2, 3, 4], 'float32'));
const expected16 = new Tensor(Float32Array.from([1, 2, 3, 4]), 'float32', [1, 4]);
if(!assertTensorEquality(actual16, expected16)) {
    throw new Error('ExpandDims Test failed');
}

// GreaterEqual Test
const actual17 = tensorTransformUtils.greaterEqual(new Tensor([1, 2, 3], 'float32'), new Tensor([2, 2, 2], 'float32'));
const expected17 = new Tensor(Uint8Array.from([0, 1, 1]), 'bool', [3]);
if(!assertTensorEquality(actual17, expected17)) {
    throw new Error('GreaterEqual Test failed');
}

// Where Test - 1D condition
const conda = new Tensor([false, false, true], 'bool');
const inp1a = new Tensor(Int32Array.from([1, 2, 3]), 'int32'); 
const inp2a = new Tensor(Int32Array.from([-1, -2, -3]), 'int32');
const actual18a = tensorTransformUtils.where(conda, inp1a, inp2a);
const expected18a = new Tensor(Int32Array.from([-1, -2, 3]), 'int32', [3]);
if(!assertTensorEquality(actual18a, expected18a)) {
    throw new Error('Where Test (a) failed');
}

// Where Test - non-1D condition
const condb = new Tensor([false, false, true, true], 'bool', [2,2]);
const inp1b = new Tensor(Int32Array.from([1, 2, 3, 4]), 'int32', [2,2]); 
const inp2b = new Tensor(Int32Array.from([-1, -2, -3, -4]), 'int32', [2,2]);
const actual18b = tensorTransformUtils.where(condb, inp1b, inp2b);
const expected18b = new Tensor(Int32Array.from([-1, -2, 3, 4]), 'int32', [2,2]);
if(!assertTensorEquality(actual18b, expected18b)) {
    throw new Error('Where Test (b) failed');
}

// Cast Test
const actual19 = tensorTransformUtils.cast(new Tensor([1.5, 2.5, 3], 'float32'), 'int32');
const expected19 = new Tensor(Int32Array.from([1, 2, 3]), 'int32', [3]);
if(!assertTensorEquality(actual19, expected19)) {
    throw new Error('Cast Test failed');
}

// Reshape Test
const actual20 = tensorTransformUtils.reshape(new Tensor(Int32Array.from([1, 2, 3, 4]), 'int32'), [2,2]);
const expected20 = new Tensor(Int32Array.from([1, 2, 3, 4]), 'int32', [2,2]);
if(!assertTensorEquality(actual20, expected20)) {
    throw new Error('Reshape Test failed');
}

// ArgMax Test(a) - 1D
const actual21a = tensorTransformUtils.argMax(new Tensor(Int32Array.from([1, 2, 3]), 'int32'));
const expected21a = new Tensor(Int32Array.from([2]), 'int32', [1]);
if(!assertTensorEquality(actual21a, expected21a)) {
    throw new Error('ArgMax Test (a) failed');
}

// ArgMax Test(b) - Non 1D
const actual21b = tensorTransformUtils.argMax(new Tensor(Int32Array.from([1, 2, 4, 3]), 'int32', [2,2]), 1);
const expected21b = new Tensor(Int32Array.from([1, 0]), 'int32', [2]);
if(!assertTensorEquality(actual21b, expected21b)) {
    throw new Error('ArgMax Test (b) failed');
}

// Max Test(a) - 1D
const actual22a = tensorTransformUtils.max(new Tensor(Int32Array.from([1, 2, 3]), 'int32'));
const expected22a = new Tensor(Int32Array.from([3]), 'int32', [1]);
if(!assertTensorEquality(actual22a, expected22a)) {
    throw new Error('Max Test (a) failed');
}

// Max Test(b) - Non 1D
const actual22b = tensorTransformUtils.max(new Tensor(Int32Array.from([1, 2, 3, 4]), 'int32', [2,2]), 1);
const expected22b = new Tensor(Int32Array.from([2, 4]), 'int32', [2]);
if(!assertTensorEquality(actual22b, expected22b)) {
    throw new Error('Max Test (b) failed');
}

console.log('All tests passed!');

// Helper for tests
function assertTensorEquality(t1: Tensor, t2: Tensor) 
{
    // type doesn't match
    if(t1.type !== t2.type) {
        return false; 
    }
    
    // dims don't match
    if(t1.dims.length !== t2.dims.length) {
        return false;
    }
    
    // dims don't match
    for(let i = 0; i < t1.dims.length; ++i) {
        if(t1.dims[i] !== t2.dims[i]) {
            return false;
        }
    }
    
    // data doesn't match
    if(t1.data.length !== t2.data.length) {
        return false;
    }

    // data doesn't match
    for(let i = 0; i < t2.data.length; ++i) {
        if(t1.data[i] !== t2.data[i]) {
            if(t1.type === 'string') {
                return false;
            }
            // number type. so allow some precision loss.
            if(Math.abs((t1.data[i] as number) - (t2.data[i] as number)) > 0.0001) {
                console.log(t1.data);
                console.log(t2.data);
                return false;
            }    
        }
    }
    
    return true;
}