var mat4 = function(){
	return new Float32Array(16);
}
{
	mat4.identity = function(output){
		output.fill(0, 0, 15);

		output[0] = 1.0;
		output[5] = 1.0;
		output[10] = 1.0;
		output[15] = 1.0;
	}

	mat4.translate = function(x, y, z, inMatrix, outMatrix){
		if(!outMatrix){
			outMatrix = inMatrix;
		}else{
			//TODO: copy data memory directly from inMatrix to out matrix;
			outMatrix[0] = inMatrix[0];
			outMatrix[1] = inMatrix[1];
			outMatrix[2] = inMatrix[2];
			outMatrix[3] = inMatrix[3];

			outMatrix[4] = inMatrix[4];
			outMatrix[5] = inMatrix[5];
			outMatrix[6] = inMatrix[6];
			outMatrix[7] = inMatrix[7];

			outMatrix[8] = inMatrix[8];
			outMatrix[9] = inMatrix[9];
			outMatrix[10] = inMatrix[10];
			outMatrix[11] = inMatrix[11];
		}

		// This is mirored across the diagonal to work properly with glsl matricies		
		outMatrix[12] = inMatrix[0] * x + inMatrix[4] * y + inMatrix[8] * z + inMatrix[12];		
		outMatrix[13] = inMatrix[1] * x + inMatrix[5] * y + inMatrix[9] * z + inMatrix[13];	
		outMatrix[14] = inMatrix[2] * x + inMatrix[6] * y + inMatrix[10] * z + inMatrix[14];		
		outMatrix[15] = inMatrix[3] * x + inMatrix[7] * y + inMatrix[11] * z + inMatrix[15];	

		return outMatrix;
	}

	mat4.rotateY = function(angleInRads, inMatrix, outMatrix){
		if(!outMatrix){
			outMatrix = inMatrix;
		}else{

			outMatrix[4] = inMatrix[4];
			outMatrix[5] = inMatrix[5];
			outMatrix[6] = inMatrix[6];
			outMatrix[7] = inMatrix[7];

			outMatrix[12] = inMatrix[8];
			outMatrix[13] = inMatrix[9];
			outMatrix[14] = inMatrix[10];
			outMatrix[15] = inMatrix[11];
		}

		var a = inMatrix[0];
		var c = inMatrix[8];
		var e = inMatrix[1];
		var g = inMatrix[9];
		var i = inMatrix[2];
		var k = inMatrix[10];
		var m = inMatrix[3];
		var o = inMatrix[11];
		var sin = Math.sin(angleInRads);
		var cos = Math.cos(angleInRads);

		outMatrix[0] = (a * cos) - (c * sin);
		outMatrix[1] = (e * cos) - (g * sin);
		outMatrix[2] = (i * cos) - (k * sin);
		outMatrix[3] = (m * cos) - (o * sin);

		outMatrix[8] = (a * sin) + (c * cos);
		outMatrix[9] = (e * sin) + (g * cos);
		outMatrix[10] = (i * sin) + (k * cos);
		outMatrix[11] = (m * sin) + (o * cos);

		return outMatrix;
	}

	// This may not be correct for opengl.
	/*mat4.multVec3 = function(inMat, inVec, outVec){
		if(!outVec){
			outVec = inVec;
		}

		var a = inVec[0];
		var b = inVec[1];
		var c = inVec[2];
		var d = inVec[3];

		outVec[0] = inMat[0] * a + inMat[1] * b + inMat[2] * c + inMat[3] * d;
		outVec[1] = inMat[4] * a + inMat[5] * b + inMat[6] * c + inMat[7] * d;
		outVec[2] = inMat[8] * a + inMat[9] * b + inMat[10] * c + inMat[11] * d;
		outVec[3] = inMat[12] * a + inMat[13] * b + inMat[14] * c + inMat[15] * d;

		return outVec;
	}*/
}

var vec3 = function(x = 0.0, y = 0.0, z = 0.0){
	var result = new Float32Array(3);
	result[0] = x;
	result[1] = y;
	result[2] = z;
	result[3] = 1.0;

	return result;
}
{
	vec3.size = function(x, y, z){
		return Math.sqrt(x * x + y * y + z * z);
	}

	vec3.distance = function(a, b){
		var x = a[0] - b[0];
		var y = a[1] - b[1];
		var z = a[2] - b[2];

		return vec3.size(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
	}

	vec3.scale = function(inputVec, scale, outputVec){
		if(!outputVec){
			outputVec = new vec3();
		}

		outputVec[0] = inputVec[0] * scale;
		outputVec[1] = inputVec[1] * scale;
		outputVec[2] = inputVec[2] * scale;

		return outputVec;
	}

	vec3.mult = function(a, b, output){
		if(!output){
			output = a;
		}

		output[0] = a[0] * b[0];
		output[1] = a[1] * b[1];
		output[2] = a[2] * b[2];

		return output;
	}

	vec3.normalize = function(inputVec, outputVec){
		if(!outputVec){
			outputVec = new vec3();
		}

		vec3.scale(inputVec, 1.0 / vec3.size(inputVec[0], inputVec[1], inputVec[2]), outputVec);

		return outputVec;
	}
}