#!/usr/bin/env python
from string import split, atof, atoi
from threading import *
from rxsrc_fade_transmit import *
from filesrc_fade_transmit import *
import time
import os

class USRPF_client_manager(Thread):

	def __init__(self):
		""" Creates a client manager that handles requests for USRP Fading simulator services.
			
		The client manager is responsible for interpreting instructions from a client,
		and if the instructions are valid, executing the commands. For each instruction, the 
		manager will provide feedback through responses.
		
		All instructions are strings that follow this format:
			
			'instruction:value'
			
			- The 'instruction' argument is an identifiable keyword:
			
				"set_recv_speed"
				"set_rf_tx_freq"
				"set_rf_tx_power"
				"set_file_sample_rate"
				"play_file"
				"play_receiving_rf"
				"play_stop"

			- The 'value' argument is optional and is dependent on the instruction keyword.
			
		All responses that the manager issues are strings that follow this format:
			
			'instruction:success:information'
			
			- The 'instrunciton' argument is essentially just an echo. This confirms that 
			  the manager has listened to what the client wants.
			- The 'success' argument can have two values: 'success' or 'failure'.
			- The 'information' argument provides additional information describing 
			  failure conditions etc.
	
		
		The manager only has two public methods, instruct() and end().
		
		Author: Jonas Hodel
		Date: 09/05/2007
			
		"""
		print "USRPF_client_manager starting up"
		# Initialize private variables.
		#------------------------------------------------------------
		self.__recv_speed = False
		self.__rf_tx_freq = None
		self.__rf_tx_power = None
		self.__file_sample_rate = None 
		self.__usrp_side = 0
		# This is a constant. Files that are sent by clients should
		# end up here.
		self.ftp_share_path = "/home/FTP-shared/upload/"  
		#---------------------------------------------------------
		
		# Setup thread control objects.
		#---------------------------------------------------------
		# Tracks whether playback thread 'run()' should start.
		# If this flag is set then play back will begin.
		self.__play_back_start = Event()
		self.__play_back_start.clear()
		# Tracks whether playback has started. If this flag is set then playback
		# has started.
		self.__play_back_started = Event()
		self.__play_back_started.clear()
		self.__stop_lock = Lock()
		#---------------------------------------------------------
		
		# This is what interacts with the USRP.
		self.__simulator = None
		Thread.__init__ ( self )
		# Makes the manager daemonic so that it terminates when the main thread exits.
		self.setDaemon(True) 
		# Starts the manager thread.
		self.start()

	def instruct(self, command_str):
		""" Instructs the manager to do things. 
			
		Instructions are a string formatted as 'instruction:value'
		After interpreting the instruction string, a response string is
		returned, in the format 'instruction:success:value'.
		
		Example:
		
			manager = USRPF_client_manager()
			reply = manager.instruct('set_rf_freq:440100000')
			
		
		See constructor docstring for further details.
						
		"""
		# Break the command string into respective parts. The expected format is
		# "instruction:value".
		(instruction, value) = split(command_str, ":")
		
		# 'play_stop' is a special instruction that is independent of the playback
		# status flag.
		if instruction == "play_stop":
			(success, information) = self.__play_stop()	
		# Other than 'play_stop', commands will only be accepted if file playback is not active.
		elif not self.__play_back_start.isSet():
			# Interprets the command associated with the received instruction.
			if instruction == "set_recv_speed":
				(success, information) = self.__set_recv_speed(atof(value))	
			elif instruction == "set_rf_tx_freq":
				(success, information) = self.__set_rf_tx_freq(atof(value))
			elif instruction == "set_rf_tx_power":
				(success, information) = self.__set_rf_tx_power(atof(value))
			elif instruction == "set_file_sample_rate":
				(success, information) = self.__set_file_sample_rate(atof(value))
			elif instruction == "play_file":
				(success, information) = self.__play_file(value)
			elif instruction == "play_receiving_rf":
				(success, information) = self.__play_receiving_rf(atof(value))
			elif instruction == "set_output_dboard":
				(success, information) = self.__set_output_dboard(atoi(value))

			else:
				success = "failure"
				information = "unrecognized command"	
		else:
			success = "failure"
			information = "USRP play back is active, this cannot be interrupted"
				
		# Forms the reply string formatted as "instruction:success:information"
		return instruction + ":" + success + ":" + information
	
	def end(self):
		""" Should be called when the manager is no longer required.
		
		This will terminate any running flow graphs and deallocate 
		the manager object. Once end has been called the manger object
		will no longer exist and will need to be recreated for further 
		use.
			
		"""
		print "USRPF_client_manager shutting down"
		if self.__simulator != None:
			# Stop the GNU Radio flow graph.
			self.__simulator.stop()
			# Invokes the destructor, killing the fading simulator.
			self.__simulator = None
		# The USRPF_client_manager dereferences itself.
		if self != None:
			self = None
	
	def run(self):
		""" This is the thread that interacts with the USRP.
		
		This thread runs indefinitely, waiting to be told to start the 
		GNU Radio flow-graph. Once the flow-graph is started the thread
		waits until the flow-graph ends either naturally or forcibly.
		Once the flow-graph has ended the thread goes back to waiting 
		for the signal to start the next flow-graph. Only one flow-graph
		can ever be active at any given time.
		
		"""
		while 1:
			# Waits for the signal to start playback (commencement of the 
			# GNU Radio flow-graph).
			self.__play_back_start.wait()
			print "Play back active"
			# Start the actual simulator.
			self.__simulator.start()
			
			# Wait while the graph isn't running.
			running = False
			while not running:
				# Returns true if running.
				running = self.__simulator.is_running()
				time.sleep(0)
			
			# Give the USRP time to stabilize (may not be required).
			time.sleep(0.2)	
			# Asserts the signal to show that the flow-graph is now active.
			self.__play_back_started.set()
			# Gives other threads a chance to run.
			time.sleep(0)			
			# This will only return once the flow graph has finished,
			self.__simulator.wait()
			self.__play_stop()
			print "Play back ended"
			## Get rid of the simulator, as it needs to be reinsatiated.
			#self.__simulator = None
			#self.__play_back_start.clear()
			#self.__play_back_started.clear()
			#print "Play back ended"
	
	def __set_recv_speed(self, recv_speed):
		""" Sets the receiver speed (affect Rayleigh fading).
		
		'recv_speed should be a float (km/h).
		If 'recv_speed' is zero then Rayleigh fading is disabled. 
		This is useful for straight through transmission from
		one of the supported sample sources (i.e. RF source, file source).
		
		Returns a tuple: success, information.
		
		"""
		# This disables Rayleigh fading				
		if recv_speed == 0:
			self.__recv_speed = recv_speed
			information = "Rayleigh fading has been disabled"
			return ("success", information)
		elif recv_speed > 0:
			self.__recv_speed = recv_speed
			information = "Receiver speed successfully set to %f km/h" % self.__recv_speed
			return ("success", information)			
		else:
			return ("failure", "Receiver speed must be greater than zero")
	
	
	def __set_output_dboard(self,usrp_side_select):
		""" Selects which side of the USRP output samples are directed to.
		
		The side select is an integer with value 0 or 1, representing A or B respectively.
		
		Returns a tuple: success, information.
		
		"""
		self.__usrp_side = usrp_side_select
		information = "USRP side %d selected" % self.__usrp_side
		return ("success", information)
			
		
	def __set_rf_tx_freq(self, rf_tx_freq):
		""" Sets the RF transmit frequency.
		
		The RF frequency input argument is a float (Hz).
		
		Returns a tuple: success, information.
		
		"""
		if rf_tx_freq >= 400e6 and rf_tx_freq <= 500e6:
			self.__rf_tx_freq = rf_tx_freq
			information = "RF Tx frequency successfully set to %f Hz" % self.__rf_tx_freq 
			return ("success", information)
		
		else:	
			return ("failure", "RF tx frequency out of range")

	def __set_rf_tx_power(self, rf_tx_power):
		""" Sets the RF transmit power.
		
		The RF power input argument is a float (dB).
		Transmit power can only roughly be set between 0 and -40 dBm.
		Outside of this range expect the unexpected.
		
		Returns a tuple: success, information.
		
		"""		
		p_min = -85
		p_max = 0
		
		if rf_tx_power >= p_min and rf_tx_power <= p_max:
			# This relationship was experimentally derived.	
			self.__rf_tx_power = pow(10.0, (rf_tx_power + 73.0) / 18.0)
			# print "self.__rf_tx_power = ", self.__rf_tx_power
			information = "RF Tx power successfully set to %f dB" % rf_tx_power
			return ("success", information)
		else:
			return ("failure", "RF tx power out of range")
			
	def __set_file_sample_rate(self, file_sample_rate):
		""" Sets the sample rate of the source file.
		
		The sample rate input argument is a float (Samples/sec).
		
		Returns a tuple: success, information.
		
		"""
		self.__file_sample_rate = file_sample_rate
		return ("success", ("File sample rate successfully set to %f samples/second" % file_sample_rate))
				
	def __play_file(self, file_name):
		""" Begin playback from a file source.
		
		'file_name' specifies which file to play. It is assumed the file is located at 
		the location defined by 'self.ftp_share_path'.
		
		Returns a tuple: success, information.
		
		"""	
		# Ensure the required parameters have been set.
		if self.__rf_tx_freq and self.__rf_tx_power and self.__file_sample_rate and self.__recv_speed is not False:
			# Test to see if the file actually exists.
			if os.path.exists(self.ftp_share_path + file_name):
				# All the required parameter values have been collected, so we are ready to 
				# actually start the fading simulator.
				self.__simulator = filesrc_fade_transmit(self.__rf_tx_freq,
													 self.__recv_speed,
													 self.__file_sample_rate, 
													 self.__rf_tx_power,  
													 self.ftp_share_path + file_name)				
				# This will signal the run() method to start running the flow-graph.
				# Once the flow-graph ends the __play_back_start flag will be cleared.
				self.__play_back_start.set()
				# Gives other threads a chance to start.
				time.sleep(0)
				# Wait until the run() method signals that playback has started.
				self.__play_back_started.wait()
				return ("success", "Play back of file has started.")
			else:
				return ("failure", "File (%s) does not exist" % file_name)
		else:
			return("failure", "Ensure that the RF transmit power, frequency, receiver speed and sample rate are set")

	def __play_receiving_rf(self, rf_rx_freq):
		""" Begin playback from an RF source.
		
		'rf_rx_freq' defines the frequency to receive on (Hz).
		
		Returns a tuple: success, information.
		
		"""		
		# Ensure the required parameters have been set.
		if self.__rf_tx_freq and self.__recv_speed is not False:
			# All the required parameter values have been collected, so we are ready to 
			# actually start the fading simulator.
			self.__simulator = rxsrc_fade_transmit(self.__rf_tx_freq, self.__recv_speed, rf_rx_freq, self.__usrp_side)				
			# This will signal the run() method to start running the flow graph.
			# Once the flow graph ends the __play_back_start will be cleared.
			self.__play_back_start.set()
			# Gives other threads a chance to start.
			time.sleep(0)
			# Wait until the run() method signals that playback has started.
			self.__play_back_started.wait()
			return ("success", "Play back of simultaneous receive has started.")
		else:
			return("failure", "Ensure that the RF transmit frequency and receiver speed have been set")
	
	# TODO > There is flaky behavior here! Calling self.__simulator.stop() 
	# should be enough to stop the flow-graph (possible bug in GNU Radio).
	def __play_stop(self):
		""" Stops playback (RF transmission).
		
		Returns a tuple: success, information.
		
		"""	
		# If all goes well these will change.
		success = "failure"
		information = "Unexpected failure"
		
		# The stopping of the flow-graph needs to be protected in a critical code block.
		# This is because a flow graph can end in two ways, 1) User calls stop, 2) the
		# sample source has no more samples. Thus, potentially this block could be 
		# re-entered before it has fully completed, which would cause errors. Hence, the code block
		# is protected by a lock to ensure that the stop operation is atomic (want be 
		# interrupted by more than one thread trying to call __play_stop()).
		self.__stop_lock.acquire()
		try:	
			if self.__play_back_start.isSet():
				# Stop the GNU Radio flow graph.
				self.__simulator.stop()

				# Wait for the flowgraph to stop running.
				running = True
				while running:
					# Returns true if running.
					running = self.__simulator.is_running()
					time.sleep(0)
				
				# Invokes the destructor, killing the fading simulator.
				self.__simulator = None
				# Clear the thread control flags to indicate that the 
				# flow graph playback has ended.
				self.__play_back_start.clear()
				self.__play_back_started.clear()
				success = "success"
				information = "Play back has stopped."
			else:
				success = "success"
				information = "Play back was already stopped."
		finally:
			self.__stop_lock.release()	
			return (success, information)		
