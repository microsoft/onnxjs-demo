import {Tensor} from 'onnxjs';
import {NumberDataType, NumberOrBoolType} from './yoloPostprocess';
import {BroadcastUtil} from './yoloPostprocessUtils';
import ndarray from 'ndarray';

export function binaryOp(
  x: Tensor, y: Tensor, opLambda: (e1: number, e2: number) => number, resultType?: NumberOrBoolType): Tensor {
    const result =
        BroadcastUtil.calc(ndarray(x.data as NumberDataType, 
                                   x.dims ? x.dims.slice(0) : [x.data.length]), 
                           ndarray(y.data as NumberDataType, 
                                   y.dims ? y.dims.slice(0) : [y.data.length]), opLambda);
    if (!result) {
      throw new Error('not broadcastable');
    }
    const rType = resultType ? resultType : x.type;
    const output = new Tensor(rType === 'bool' ? Uint8Array.from(result.data) : result.data as NumberDataType,
        rType, result.shape);
    return output;
}