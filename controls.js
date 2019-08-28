var ControlBuffer = BaseObject("ControlBuffer", false);
{
	function initControls(canvas){
		canvas.addEventListener('keydown', (event) => {
			const keyName = event.key;

			
		});
	}

	ControlBuffer.prototype.init = function(canvas) {
		// FIFO Stack

		this.controls = {
			"FORWARD" : 38,
			"BACKWARD" : 40,
			"STRIFE_LEFT" : 37,
			"STRIFE_RIGHT" : 39
		}

		this.keys = new Uint8Array(256);
		this.buffer = new Uint8Array(8);
		this.headLocation = 0;
		this.count = 0;

		initControls.call(this, canvas);
	};

	ControlBuffer.prototype.push = function(action){
		var location = this.headLocation + this.count;

		while(location >= this.buffer.length){
			location -= this.buffer.length;
		}

		this.buffer[location] = action;
		++this.count;
	};

	ControlBuffer.prototype.pop = function(action){
		var result = this.buffer[this.headLocation];
		++this.headLocation;

		if(this.headLocation >= this.buffer.length){
			this.headLocation -= this.buffer.length;
		}

		--this.count;

		return result;
	};
}