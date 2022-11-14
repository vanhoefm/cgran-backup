#!/usr/bin/python

from  gnuradio import gr;

# arbitrary f1 to f2 resampler
class resampler_cc(gr.hier_block2):
    def __init__(self,
                    input_rate,
                    output_rate,
                    max_taps = 200):

        gr.hier_block2.__init__(self, "resampler",
                                    gr.io_signature(1,1,gr.sizeof_gr_complex),
                                    gr.io_signature(1,1,gr.sizeof_gr_complex))

        blks = [];
    
        blks.append(self);
        
        R = output_rate*1.0/input_rate;
        interp = 1;
        while(R > 1):
            interp = interp + 1;
            R = (output_rate*1.0)/(input_rate*interp);
        
        lpf_cutoff_1 = 1.0/interp;
        #print "lpf_1 = %f"%(lpf_cutoff_1);

        lpf_cutoff_2 = output_rate*1.0/(input_rate*interp);
        #print "lpf_2 = %f"%(lpf_cutoff_2);

        lpf = min(lpf_cutoff_1, lpf_cutoff_2);
        #print "lpf = %f\n"%(lpf);

        decim = interp*1.0*input_rate/output_rate;
    
        trans = 0.0001;
        taps = gr.firdes.low_pass(1, 1, lpf/2.0, 0.001 );
        while(len(taps) > max_taps):
            trans = trans*2;
            taps = gr.firdes.low_pass(1, 1, lpf/2.0, trans );
            #print "trans = %f, taps = %d"%(trans, len(taps));
     
        # add interpolator and filter   
        blks.append( gr.interp_fir_filter_ccf(interp, taps) );

        # add frational interpolated decimation       
        blks.append(gr.fractional_interpolator_cc( 0, decim ) );

        lpf_freq = min( output_rate/input_rate, R );
        #print "R = %f (interp = %d)"%(R,interp)
        
        blks.append(self);
 
        # connect up blocks and go   
        for i in range(0,len(blks)-1):
            self.connect(blks[i], blks[i+1]);



