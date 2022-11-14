#
# Copyright 2011 Anton Blad, Borja Martinez.
# 
# This file is part of OpenRD
# 
# OpenRD is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3, or (at your option)
# any later version.
# 
# OpenRD is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#

import math
from gnuradio import gr
from openrd import pr, dmod, pyblk

class channel(gr.hier_block2):
	def __init__(self, name = "channel"):
		gr.hier_block2.__init__(self, name,
				gr.io_signature(1, 1, gr.sizeof_gr_complex), 
				gr.io_signature(1, 1, gr.sizeof_gr_complex))

class channel_ideal(channel):
	def __init__(self):
		channel.__init__(self, "channel_ideal")

		null = gr.null_source(gr.sizeof_gr_complex)
		add = gr.add_cc() 
		
		self.connect(self, (add, 0))
		self.connect(null, (add, 1))
		self.connect(add, self)

class channel_rayleigh(channel):
	def __init__(self, modulation, fc, sample_rate, samples_per_symbol = 16, speed = 0, db=-10000):
		channel.__init__(self, "channel_rayleigh")

		#####################################FADING VARIABLES####################################
		self.fc = fc
		M=8
		N=4*M+2
		lambd=3e8/fc	# we suppose a carrier frequency of 10 Mhz and 3e8 m/s
		fd=speed/lambd
		pi=math.pi
		wd=2*pi*fd
		ATTENFACTOR=2/math.sqrt(N)
		ATTENFACTORUs=complex(0,1)*2/math.sqrt(N) # complex(0,1)=j
		wn=[]
		beta=[]
		b=[]
		a=[]
		samplerate = sample_rate/samples_per_symbol
		self.samplerate = samplerate
		########################END OF  FADING VARIABLES #############################################

		self.gr_add_cc_0 = gr.add_vcc(1)

		#########################################CALCULUS OF THE NOISE POWER DUE TO THE SNR and MAXIMUX POWER SIGNAL######
		sigmadeviance=math.sqrt(0.5*pow(10,db/10.0))

		# Create the noise source that will implemente the channel
		self.gr_noise_source_c=gr.noise_source_c(gr.GR_GAUSSIAN,sigmadeviance,42)
		#################################################################################
		
		##################### generate the coefficients #############################################################
		for i in range(1,M+2):
			if i< M+1:
				temp=math.cos((2*pi*i)/N)
				wn.append(wd*temp)
				beta.append((pi*i)/M)
				b.append(2*math.sin(beta[i-1]))
				a.append(2*math.cos(beta[i-1]))

			else:
				wn.append(wd)
				beta.append(pi/4)
				b.append(math.sqrt(2)*math.sin(beta[i-1]))
				a.append(math.sqrt(2)*math.cos(beta[i-1]))	
		# Here I have all the coefficients for [beta wn b a]
		#####################end of COEFFICIENTS#############################################################

		#########################GENERATION OF THE COMPONENTS FOR THE FADING BLOCK######
			##############UC(T) just cos with different frequencies and amplitudes####################
		self.UC1=gr.sig_source_f(samplerate,gr.GR_COS_WAVE,2*pi*wn[0],a[0],0)	#Uc=sum(an*cos(wn*t))
		self.UC2=gr.sig_source_f(samplerate,gr.GR_COS_WAVE,2*pi*wn[1],a[1],0)
		self.UC3=gr.sig_source_f(samplerate,gr.GR_COS_WAVE,2*pi*wn[2],a[2],0)
		self.UC4=gr.sig_source_f(samplerate,gr.GR_COS_WAVE,2*pi*wn[3],a[3],0)
		self.UC5=gr.sig_source_f(samplerate,gr.GR_COS_WAVE,2*pi*wn[4],a[4],0)
		self.UC6=gr.sig_source_f(samplerate,gr.GR_COS_WAVE,2*pi*wn[5],a[5],0)
		self.UC7=gr.sig_source_f(samplerate,gr.GR_COS_WAVE,2*pi*wn[6],a[6],0)
		self.UC8=gr.sig_source_f(samplerate,gr.GR_COS_WAVE,2*pi*wn[7],a[7],0)
		self.UC8=gr.sig_source_f(samplerate,gr.GR_COS_WAVE,2*pi*wn[7],a[7],0)
		self.UC9=gr.sig_source_f(samplerate,gr.GR_COS_WAVE,2*pi*wn[8],a[8],0)
			############## END OF UC(T)#########################################
			#################US(T) just sins with different frequencies and amplitudes ##############

		self.US1=gr.sig_source_f(samplerate,gr.GR_COS_WAVE,2*pi*wn[0],b[0],0)	#US=sum(bn*sin(wn*t))
		self.US2=gr.sig_source_f(samplerate,gr.GR_COS_WAVE,2*pi*wn[1],b[1],0)
		self.US3=gr.sig_source_f(samplerate,gr.GR_COS_WAVE,2*pi*wn[2],b[2],0)
		self.US4=gr.sig_source_f(samplerate,gr.GR_COS_WAVE,2*pi*wn[3],b[3],0)
		self.US5=gr.sig_source_f(samplerate,gr.GR_COS_WAVE,2*pi*wn[4],b[4],0)
		self.US6=gr.sig_source_f(samplerate,gr.GR_COS_WAVE,2*pi*wn[5],b[5],0)
		self.US7=gr.sig_source_f(samplerate,gr.GR_COS_WAVE,2*pi*wn[6],b[6],0)
		self.US8=gr.sig_source_f(samplerate,gr.GR_COS_WAVE,2*pi*wn[7],b[7],0)
		self.US8=gr.sig_source_f(samplerate,gr.GR_COS_WAVE,2*pi*wn[7],b[7],0)
		self.US9=gr.sig_source_f(samplerate,gr.GR_COS_WAVE,2*pi*wn[8],b[8],0)
			################## END OF US(T) ############################################################
			
		#########################I  create the adders for both UC AND US I need 6 adders for each ###########
			######## UC(T) ADDERS########
		self.adUC1=gr.add_ff()
		self.adUC2=gr.add_ff()
		self.adUC3=gr.add_ff()
		self.adUC4=gr.add_ff()
		self.adUC5=gr.add_ff()
		self.adUC6=gr.add_ff()
		self.adUC7=gr.add_ff()
		self.adUC8=gr.add_ff()
			#and last multiplier
		self.converterUC =gr.float_to_complex()
		self.mulUC=gr.multiply_const_cc(ATTENFACTOR)
			########END OF UC(T) ADDERS#####
			######## US(T) ADDERS #########			
		self.adUS1=gr.add_ff()
		self.adUS2=gr.add_ff()
		self.adUS3=gr.add_ff()
		self.adUS4=gr.add_ff()
		self.adUS5=gr.add_ff()
		self.adUS6=gr.add_ff()
		self.adUS7=gr.add_ff()
		self.adUS8=gr.add_ff()
			#last multiplier
		self.converterUS =gr.float_to_complex()
		self.mulUS=gr.multiply_const_cc(ATTENFACTORUs)
			########END OF US(T) ADDERS ######
		
		######################### end of adders for both UC and US ##########################################
			########LAST STEP ADD BOTH Uc(t)+j*Us(t)#########################
		self.adfinal=gr.add_cc()
			#######done##########################################
			
		############### END OF GENERATION OF components FOR THE FADING BLOCK#############
		self.mult=gr.multiply_vcc()		# I use this multiplier to multiply the fading filter with the sinwave
		####################### conection of the fading block ####################################################

		#####################Uc(t) part ######################################################
		self.connect(self.UC1,(self.adUC1,0))
		self.connect(self.UC2,(self.adUC1,1))
		self.connect(self.adUC1,(self.adUC2,0)) # THE EXIT OF THE FIRST ADDER IS THE INPUT OF THE SECOND ADDER
		self.connect(self.UC3,(self.adUC2,1))
		self.connect(self.adUC2,(self.adUC3,0))	 # THE EXIT OF THE SECOND ADDER IS THE INPUT OF THE THIRD ADDER	
		self.connect(self.UC4,(self.adUC3,1))
		self.connect(self.adUC3,(self.adUC4,0)) # THE EXIT OF THE THIRD ADDER IS THE INPUT OF THE FOURTH ADDER
		self.connect(self.UC5,(self.adUC4,1))
		self.connect(self.adUC4,(self.adUC5,0)) # THE EXIT OF THE FOURTH ADDER IS THE INPUT OF THE FIFTH ADDER
		self.connect(self.UC6,(self.adUC5,1))
		self.connect(self.adUC5,(self.adUC6,0)) # THE EXIT OF THE FIFTH ADDER IS THE INPUT OF THE SIXTH ADDER
		self.connect(self.UC7,(self.adUC6,1))
		self.connect(self.adUC6,(self.adUC7,0))
		self.connect(self.UC8,(self.adUC7,1))
		self.connect(self.adUC7,(self.adUC8,0))
		self.connect(self.UC9,(self.adUC8,1))
			#### NOW WE HAVE TO MULTIPLY THE OUTPUT BY 2/SQRT(N) remember
		self.connect(self.adUC8,self.converterUC,self.mulUC)
		######################## end of Uc(t) part############################################################
			#####################Us(t) part ######################################################
		self.connect(self.US1,(self.adUS1,0))
		self.connect(self.US2,(self.adUS1,1))
		self.connect(self.adUS1,(self.adUS2,0)) # THE EXIT OF THE FIRST ADDER IS THE INPUT OF THE SECOND ADDER
		self.connect(self.US3,(self.adUS2,1))
		self.connect(self.adUS2,(self.adUS3,0))	 # THE EXIT OF THE SECOND ADDER IS THE INPUT OF THE THIRD ADDER	
		self.connect(self.US4,(self.adUS3,1))
		self.connect(self.adUS3,(self.adUS4,0)) # THE EXIT OF THE THIRD ADDER IS THE INPUT OF THE FOURTH ADDER
		self.connect(self.US5,(self.adUS4,1))
		self.connect(self.adUS4,(self.adUS5,0)) # THE EXIT OF THE FOURTH ADDER IS THE INPUT OF THE FIFTH ADDER
		self.connect(self.US6,(self.adUS5,1))
		self.connect(self.adUS5,(self.adUS6,0)) # THE EXIT OF THE FIFTH ADDER IS THE INPUT OF THE SIXTH ADDER
		self.connect(self.US7,(self.adUS6,1))
		self.connect(self.adUS6,(self.adUS7,0))
		self.connect(self.US8,(self.adUS7,1))
		self.connect(self.adUS7,(self.adUS8,0))
		self.connect(self.US9,(self.adUS8,1))
			#### NOW WE HAVE TO MULTIPLY THE OUTPUT BY 2/SQRT(N) remember
		self.connect(self.adUS8,self.converterUS,self.mulUS)
			##########################end of Us(t) part####################################################
			####NOW I add both parts Uc(t)+j*Us(t)######################################################
		self.connect(self.mulUC,(self.adfinal,0))
		self.connect(self.mulUS,(self.adfinal,1))
		###########################end of connection of the fading block###########################################

		self.rec = pyblk.receiver_cc(modulation, samples_per_symbol = 16)
		self.xmit = pyblk.transmitter_cc(modulation, samples_per_symbol = 16)

		##############################################
		self.connect(self,self.rec, (self.mult,0))
		self.connect(self.adfinal,(self.mult,1))
		self.connect(self.mult,(self.gr_add_cc_0, 0))
		self.connect(self.gr_noise_source_c,(self.gr_add_cc_0, 1))
		self.connect(self.gr_add_cc_0,self.xmit,self) # I have to add self in the end since this is the output of the fading hierarchical block also in the beggining

	def set_speed(self,speed=0):	# This class would set the speed of the vehicle it can be changed anytime
		# here I must be able to change the speed of the vehicle above
		pi = math.pi
		lambd=3e8/self.fc
		fd=speed/lambd
		wd=2*pi*fd
		#####################################FADING VARIABLES####################################
		M=8
		N=4*M+2
		wn=[]
		########################END OF  FADING VARIABLES #############################################
		########################### to have the values of wn again########################################
		for i in range(1,M+2):
			if i< M+1:
				temp=math.cos((2*pi*i)/N)
				wn.append(wd*temp)
			else:
				wn.append(wd)

		# Here I have all the coefficients for [beta wn b a]
		#####################end of COEFFICIENTS#############################################################
		self.UC1.set_frequency(2*pi*wn[0]) #wn[0] is not defined in this function watch out
		self.UC2.set_frequency(2*pi*wn[1])
		self.UC3.set_frequency(2*pi*wn[2])
		self.UC4.set_frequency(2*pi*wn[3])
		self.UC5.set_frequency(2*pi*wn[4])
		self.UC6.set_frequency(2*pi*wn[5])
		self.UC7.set_frequency(2*pi*wn[6])
		self.UC8.set_frequency(2*pi*wn[7])
		self.UC9.set_frequency(2*pi*wn[8])

		self.US1.set_frequency(2*pi*wn[0]) #wn[0] is not defined in this function watch out
		self.US2.set_frequency(2*pi*wn[1])
		self.US3.set_frequency(2*pi*wn[2])
		self.US4.set_frequency(2*pi*wn[3])
		self.US5.set_frequency(2*pi*wn[4])
		self.US6.set_frequency(2*pi*wn[5])
		self.US7.set_frequency(2*pi*wn[6])
		self.US8.set_frequency(2*pi*wn[7])
		self.US9.set_frequency(2*pi*wn[8])
		
	def set_noise(self,db):
		self.gr_noise_source_c.set_amplitude(math.sqrt(0.5*pow(10,db/10.0)))

class channel_awgn(channel):	
	def __init__(self, modulation, samples_per_symbol, seed=0, db=-1000):
		channel.__init__(self, "channel_awgn")

		self.d_modulation = modulation

		self.rec = dmod.receiver_cc(modulation, samples_per_symbol = samples_per_symbol)
		self.awgn = gr.noise_source_c(gr.GR_GAUSSIAN, 0, seed)
		self.add = gr.add_cc()
		self.xmit = dmod.transmitter_cc(modulation, samples_per_symbol = samples_per_symbol)

		self.set_noise(db)

		self.connect(self, self.rec, (self.add, 0))
		self.connect(self.awgn, (self.add, 1))
		self.connect(self.add, self.xmit, self)

	def set_noise(self, db):
		sigma = math.sqrt(0.5*pow(10,db/10.0))
		self.awgn.set_amplitude(sigma)

