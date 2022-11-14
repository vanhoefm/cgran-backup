/* -*- c++ -*- */
/*
 * Copyright 2007 Free Software Foundation, Inc.
 * 
 * This file is part of GNU Radio
 * 
 * GNU Radio is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3, or (at your option)
 * any later version.
 * 
 * GNU Radio is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <gmsk.h>

static bool verbose = false;
static bool demod_debug = false;
  
std::ofstream cf_ofile;
std::ofstream fm_ofile;
std::ofstream cr_ofile;
std::ofstream sl_ofile;
std::ofstream corr_ofile;

bool d_squelch;

long t_samples;

gmsk::gmsk(mb_runtime *rt, const std::string &instance_name, pmt_t user_arg)
  : mb_mblock(rt, instance_name, user_arg),
  d_samples_per_symbol(SAMPLES_PER_SYMBOL),
  d_bt(0.35),
  d_gain_mu(0.175),
  d_mu(0.5),
  d_freq_error(0.0),
  d_omega_relative_limit(0.005),
  d_amplitude(12000),
  d_low_pass(false),
  d_corr_thresh(12),
  d_fmdemod_last(0),
  d_disk_write(false),
  d_nframes_recvd(0)
{
  if(d_disk_write) {
  cf_ofile.open("chanf.dat", std::ios::binary|std::ios::out);
  fm_ofile.open("fmdemod.dat", std::ios::binary|std::ios::out);
  cr_ofile.open("clock_recovery.dat", std::ios::binary|std::ios::out);
  sl_ofile.open("slicer.dat", std::ios::binary|std::ios::out);
  corr_ofile.open("corr.dat", std::ios::binary|std::ios::out);
  }

  d_squelch = true;

  t_samples =0;

  if (pmt_is_dict(user_arg)) {

    if(pmt_t interp = pmt_dict_ref(user_arg, pmt_intern("interp-tx"), PMT_NIL)) {
      if(!pmt_eqv(interp, PMT_NIL)) {
        d_usrp_interp = pmt_to_long(interp);
      } else {
        std::cout << "[GMSK] Failure: need to specify interp-tx\n";
        shutdown_all(PMT_F);
        return;
      }
    }
    
    if(pmt_t decim = pmt_dict_ref(user_arg, pmt_intern("decim-rx"), PMT_NIL)) {
      if(!pmt_eqv(decim, PMT_NIL)) {
        d_usrp_decim = pmt_to_long(decim);
      } else {
        std::cout << "[GMSK] Failure: need to specify decim-rx\n";
        shutdown_all(PMT_F);
        return;
      }
    }
  }

  // Initialize the ports
  define_ports();

  if(verbose)
    std::cout << "[GMSK] Initializing....\n";
  
  // Initialize gmsk
  d_ntaps = 4 * d_samples_per_symbol;
  d_sensitivity = (M_PI / 2.0) / (double)d_samples_per_symbol;
  
  // Get a handle to a block to convert to NRZ
  d_bts = gr_make_bytes_to_syms();

  d_gaussian_taps = gr_firdes::gaussian(
      1,                    // gain
      d_samples_per_symbol,
      d_bt,
      d_ntaps);

  // rectangular window
  for(int i = 0; i < d_samples_per_symbol; i++)
    d_sqwave.push_back(1);

  // calculate taps, need to allocate size of the two vectors-1
  for(int i=0; i < (int)(d_sqwave.size() + d_gaussian_taps.size() - 1); i++)
    d_taps.push_back(0);
  convolve(&d_gaussian_taps[0], 
           &d_sqwave[0], 
           &d_taps[0], 
           d_gaussian_taps.size(),
           d_sqwave.size());

  if(verbose) {
    std::cout << "[GMSK] taps: ";
    for(int j=0; j < (int)d_taps.size(); j++) {
      std::cout << d_taps[j] << " ";
    }
    std::cout << std::endl;
  }
  
  // Gaussian filter
  d_gf = gr_make_interp_fir_filter_fff(d_samples_per_symbol, d_taps);

  // FM modulation
  d_fm_mod = gr_make_frequency_modulator_fc(d_sensitivity);

  ////// DEMOD SETUP //////
  
  // low pass
  d_sw_decim = 1;
  d_chan_coeffs = gr_firdes::low_pass(1.0,  //gain
                                      d_sw_decim * d_samples_per_symbol,
                                      1.0,
                                      0.5,
                                      gr_firdes::WIN_HANN);

  if(verbose) 
  std::cout << "[CCC] taps: ";
  for(int h=0; h < (int)d_chan_coeffs.size(); h++) {
    if(verbose)
      std::cout << d_chan_coeffs[h] << " ";
    d_cchan_coeffs.push_back(d_chan_coeffs[h]);
  }
  if(verbose)
    std::cout << std::endl;
  
  d_chan_filt = gr_make_fft_filter_ccc(d_sw_decim, d_cchan_coeffs);

  
  // FM demodulation
  d_fm_demod = gr_make_quadrature_demod_cf(1.0 / d_sensitivity);

  // Mueller and Müller (M&M) discrete-time error-tracking synchronizer
  d_omega = d_samples_per_symbol*(1+d_freq_error);
  d_gain_omega = 0.25 * d_gain_mu * d_gain_mu;
  d_cr = gr_make_clock_recovery_mm_ff(d_omega, 
                                      d_gain_omega,
                                      d_mu,
                                      d_gain_mu,
                                      d_omega_relative_limit);

  d_slicer = gr_make_binary_slicer_fb();

  // Setup the correlation block (bit sequence finder). 
  d_preamble = PREAMBLE;                // Preamble to help synchronization
  d_access_code = FRAMING_BITS;         // Framing bits to detect frame
  d_postamble = POSTAMBLE;              // Post amble to pad end of frame
  d_corr = gr_make_correlate_access_code_bb(d_access_code, d_corr_thresh);

  if(verbose)
    std::cout << "[GMSK] Initialized\n";
}

gmsk::~gmsk()
{
  if(d_disk_write) {
    fm_ofile.close();
    cr_ofile.close();
    sl_ofile.close();
    cf_ofile.close();
  }
}

// The full functionality of CMAC is based on messages passed back and forth
// between the application and a physical layer and/or usrp_server.  Each
// message triggers additional events, states, and messages to be sent.
void gmsk::handle_message(mb_message_sptr msg)
{

  // The MAC functionality is dispatched based on the event, which is the
  // driving force of the MAC.  The event can be anything from incoming samples
  // to a message to change the carrier sense threshold.
  pmt_t event = msg->signal();
  pmt_t data = msg->data();
  pmt_t port_id = msg->port_id();

  pmt_t handle = PMT_F;
  pmt_t status = PMT_F;
  pmt_t dict = PMT_NIL;
  std::string error_msg;

  if(pmt_eq(event, s_cmd_mod)
      && pmt_eq(d_cs->port_symbol(), port_id)) {
      mod(data);
  }
  
  if(pmt_eq(event, s_cmd_demod)
      && pmt_eq(d_cs->port_symbol(), port_id)) {
      demod(data);
  }

}

void gmsk::mod(pmt_t data) {
  
  size_t n_bytes;

  // Frame properties
  pmt_t invocation_handle = pmt_nth(0, data);
  const void *payload = pmt_uniform_vector_elements(pmt_nth(1, data), n_bytes);
  pmt_t pkt_properties = pmt_nth(2, data);

  // Access code and pre-amble
  std::vector<unsigned char>    b_access_code(d_access_code.length()/8);
  conv_to_binary(d_access_code, b_access_code);
  std::vector<unsigned char>    b_preamble(d_preamble.length()/8);
  conv_to_binary(d_preamble, b_preamble);
  std::vector<unsigned char>    b_postamble(d_postamble.length()/8);
  conv_to_binary(d_postamble, b_postamble);

  // Take data from file
  std::ofstream nrz_ofile;
  std::ofstream gf_ofile;
  std::ofstream fm_ofile;

  // The full premod data is the preamble (to help with clock synchronization),
  // the start of frame bits (access code), the frame header, and the payload.
  long dsize = b_preamble.size()+b_access_code.size()+n_bytes+b_postamble.size();
  unsigned char *full_premod_data = (unsigned char *) calloc(1, dsize);

  // Copy in everything
  memcpy(full_premod_data, &b_preamble[0], 1);
  memcpy(full_premod_data+b_preamble.size(), &b_access_code[0], b_access_code.size());
  memcpy(full_premod_data+b_preamble.size()+b_access_code.size(), payload, n_bytes);
  memcpy(full_premod_data+b_preamble.size()+b_access_code.size()+n_bytes, &b_postamble[0], 1);

  // Where to store the NRZ output
  long bts_nout = dsize*BITS_PER_BYTE;
  std::vector<float>        bts_output(bts_nout);
  std::vector<const void*>  bts_pinput(1, full_premod_data);
  std::vector<void*>        bts_poutput(1, &bts_output[0]);
  d_bts->work(bts_nout, bts_pinput, bts_poutput);
  if(d_disk_write) {
    nrz_ofile.open("nrz.dat", std::ios::binary|std::ios::out);
    nrz_ofile.write((const char *)&bts_output[0], bts_output.size()*4);
    nrz_ofile.close();
  }

  int gf_fir_history_size = d_ntaps/d_samples_per_symbol;
  int gf_tinput = bts_output.size() + gf_fir_history_size;
  float *fir_hack = (float *)calloc16Align(1, gf_tinput*sizeof(float));

  // Insert the FIR history, if there is none the first gf_fir_history_size items
  // will be 0 (calloc zeros memory).
  if(d_gf_history.size()!=0) 
    for(int i=0; i<gf_fir_history_size; i++)
      fir_hack[i] = d_gf_history[i];

  // Now, insert the rest starting after the history offset
  for(int i=0; i<(int)bts_output.size(); i++)
    fir_hack[i+gf_fir_history_size] = bts_output[i];

  // Now we pass the data on to the Gaussian filter
  long gf_nout = d_samples_per_symbol * bts_nout;
  std::vector<float>        gf_output(gf_nout);
  std::vector<const void*>  gf_pinput(1, &fir_hack[0]);
  std::vector<void*>        gf_poutput(1, &gf_output[0]);
  d_gf->work(gf_nout, gf_pinput, gf_poutput);
  if(d_disk_write) {
    gf_ofile.open("gaussian_filter.dat", std::ios::binary|std::ios::out);
    gf_ofile.write((const char *)&gf_output[0], gf_output.size()*4);
    gf_ofile.close();
  }

  free16Align(fir_hack);

  // Save the last items as history ... this is really unnecessary as we filter
  // the whole data at once.
//  fir_hack.clear();
//  for(int i=0; i<gf_fir_history_size; i++)
//    fir_hack.push_back(bts_output[bts_output.size()-1-i]);

  // Finally, FM modulate it
  long fm_nout = gf_nout;
  std::vector<gr_complex>   fm_output(fm_nout);
  std::vector<const void*>  fm_pinput(1, &gf_output[0]);
  std::vector<void*>        fm_poutput(1, &fm_output[0]);
  d_fm_mod->work(fm_nout, fm_pinput, fm_poutput);
  if(d_disk_write) {
    fm_ofile.open("fmmod.dat", std::ios::binary|std::ios::out);
    fm_ofile.write((const char *)&fm_output[0], fm_output.size()*8);
    fm_ofile.close();
  }

  // Convert from 64 bit I/Q to 32 bit I/Q and fill in vector
  size_t ignore;
  pmt_t v_mod_data = pmt_make_s16vector(fm_output.size()*2, 0);   // 16-bits for each I and Q 
  int16_t *mod_data = pmt_s16vector_writable_elements(v_mod_data, ignore);
  for(int i=0; i < (int)fm_output.size(); i++) {
    mod_data[2*i] =   (int16_t) (fm_output[i].real() * d_amplitude);
    mod_data[2*i+1] = (int16_t) (fm_output[i].imag() * d_amplitude);
  }

  //demod(pmt_list3(PMT_NIL, PMT_NIL, v_mod_data));

  // Send back the modulated data with the properties
  d_cs->send(s_response_mod,
             pmt_list3(invocation_handle,
                       v_mod_data,
                       pkt_properties));
                       
  if(verbose)
    std::cout << "[GMSK] Modulated data\n";
}

//void gmsk::demod(const std::vector<gr_complex> mod_data)
void gmsk::demod(pmt_t data)
{
  size_t n_bytes;

  pmt_t invocation_handle = pmt_nth(0, data);
  
  const void *mod_data = pmt_uniform_vector_elements(pmt_nth(2, data), n_bytes);
  int16_t *samples = (int16_t *)mod_data;
  unsigned long timestamp = (unsigned long)pmt_to_long(pmt_nth(3, data));
  pmt_t demod_properties = pmt_nth(5,data);

  if(demod_debug)
    std::cout << "[GMSK] Demodulating (" << n_bytes/4 << ")...";

  // Store the samples as complex, I=16bits Q=16bits
  // Take from the samples queue first
  long swaiting = d_filterq.size();
  std::vector<gr_complex>   c_samples((n_bytes/4)+swaiting);
  for(int h=0; h<(int)swaiting; h++) {
    c_samples[h] = d_filterq.front();
    d_filterq.pop();
  }

  // Now take samples from the incoming data, do a little power squelching while
  // we're at it
  const long SQUELCH = 100;
  long stored = 0;
  bool found=false;
  for(int j=0; j<(int)c_samples.size()-swaiting; j++) {
    if(d_squelch && !found) {
      if(sqrt(samples[j*2]*samples[j*2]+samples[j*2+1]*samples[j*2+1]) > SQUELCH) {
        c_samples[stored+swaiting] = gr_complex(samples[j*2], samples[j*2+1]);
        stored++;
        found=true;
      }
    } else {
      c_samples[stored+swaiting] = gr_complex(samples[j*2], samples[j*2+1]);
      stored++;
    }
  }

  long c_tsamples = stored + swaiting;

  // Push the extra samples on to the queue (input has to be % 20)
  long cf_nout = c_tsamples - (c_tsamples % 20);
  for(int k=cf_nout; k<(int)c_tsamples; k++)
    d_filterq.push(c_samples[k]);

  // Need to bail if not enough samples for the filter
  if(cf_nout==0) {
    if(demod_debug) 
      std::cout << std::endl;
    return;
  }

  t_samples += cf_nout;

  // Go through lowpass chan filter
  std::vector<gr_complex>   cf_output(cf_nout);
  std::vector<const void*>  cf_pinput(1, &c_samples[0]);
  std::vector<void*>        cf_poutput(1, &cf_output[0]);
  d_chan_filt->work(cf_nout, cf_pinput, cf_poutput);
  if(d_disk_write) {
    cf_ofile.write((const char *)&cf_output[0], cf_output.size()*4);
  }
  if(demod_debug) {
    std::cout << " CF ";
    fflush(stdout);
  }
  
  int fm_history_size = 1;
  int fm_tinput = cf_output.size()+1;
  gr_complex *fm_fir_hack  = (gr_complex *)calloc16Align(1, fm_tinput*sizeof(gr_complex));

  // Insert the FIR history, if there is none the first fm_history_size will be 0
  if(d_fm_history.size()!=0)
    for(int i=0; i<fm_history_size; i++)
      fm_fir_hack[i] = d_fm_history[i];

  // Insert the rest now
  for(int i=0; i<(int)cf_output.size(); i++)
    fm_fir_hack[i+fm_history_size] = cf_output[i];

  // Pass through the FM
  long fm_nout = cf_output.size();
  std::vector<float>        fm_output(fm_nout);
  std::vector<const void*>  fm_pinput(1, &fm_fir_hack[0]);
  std::vector<void*>        fm_poutput(1, &fm_output[0]);
  d_fm_demod->work(fm_nout, fm_pinput, fm_poutput);
  if(d_disk_write) {
    fm_ofile.write((const char *)&fm_output[0], fm_output.size()*4);
  }

  free16Align(fm_fir_hack);

  // Place the last input into our history
  d_fm_history.clear();
  for(int i=0; i<fm_history_size; i++)
    d_fm_history.push_back(cf_output[cf_output.size()-1-i]);

  if(demod_debug) {
    std::cout << " FM ";
    fflush(stdout);
  }
  
  // Perform clock recovery and split
  //
  // This is a bit trickier, we have modified the clock recovery method to
  // return the number of input items actually consumed and compute the output
  // from the input. 
  gr_vector_int     cr_in(1);
  std::vector<float> cr_fullout;
    
  // The input is what is left over from the last CR consumption plus the fm
  // output
  long cr_csize = d_crq.size();   // size of items left in input buffer
  int cr_tinput = cr_csize + fm_output.size();  // total_input = cr_csize + new_input

  // 16-byte aligned input needed for optimizations
  float *cr_input = (float *)calloc16Align(1, cr_tinput*sizeof(float));

  // Push what was left over from the previous consume() into the new input buffer
  for(int p=0; p<(int)cr_csize; p++) {
    cr_input[p] = d_crq.front();
    d_crq.pop();
  }

  // Push the new input (from the FM modulation block) into the input buffer
  for(int f=0; f<(int)cr_tinput-cr_csize; f++)
    cr_input[f+cr_csize] = fm_output[f];

  // Total number consumed to shift the input pointer
  long cr_tconsumed = 0;
  long cr_nout;

  while(1) {

    // Start at 2^13 and probe until the forecast < input
    cr_nout=8192;
    while(cr_nout>1) {
      d_cr->forecast(cr_nout, cr_in);

      if(cr_in[0]<=(cr_tinput-cr_tconsumed))
        break;
      
      cr_nout/=2;
    }

    // Not enough, break free
    if(cr_in[0]>(cr_tinput-cr_tconsumed-6)) {
      for(int i=cr_tconsumed; i<cr_tinput; i++)
        d_crq.push(cr_input[i]);
      if(demod_debug)
        std::cout << "\n  Not enough... " << cr_in[0] << " > " << (cr_tinput-cr_tconsumed) << " to produce " << cr_nout << std::endl;
      break;
    }

    if(demod_debug) {
      std::cout << "\n  nout_in(" << cr_nout << "," << cr_in[0] << "," << (cr_tinput-cr_tconsumed) << ") ";
      fflush(stdout);
    }
    
    cr_in[0] = cr_tinput-cr_tconsumed;
    

    //std::vector<float>        cr_output(cr_nout);
    float *cr_output = (float *)calloc16Align(1, cr_nout*sizeof(float));
    std::vector<const void*>  cr_pinput(1, &cr_input[cr_tconsumed+0]);
    std::vector<void*>        cr_poutput(1, &cr_output[0]);
    long consumed = d_cr->general_work(cr_nout, cr_in, cr_pinput, cr_poutput);
    if(d_disk_write) {
      cr_ofile.write((const char *)&cr_output[0], cr_nout*4);
      cr_ofile.flush();
    }
    if(demod_debug) {
      std::cout << "CR(" << cr_csize
//                << "," << fm_output.size()
                << "," << cr_in[0] 
                << "," << consumed 
                << "," << cr_nout << ") ";
      fflush(stdout);
    }

    assert(consumed>=0);
//    for(int d=consumed; d<cr_in[0]; d++)
//      d_crq.push(cr_input[d]);
    
    cr_tconsumed+=consumed;

    for(int i=0; i<cr_nout; i++)
      cr_fullout.push_back(cr_output[i]);

    free16Align(cr_output);
  }

  free16Align(cr_input);

  // Slice
  long sl_nout = cr_fullout.size();
  std::vector<unsigned char>  sl_output(sl_nout);
  std::vector<const void*>    sl_pinput(1, &cr_fullout[0]);
  std::vector<void*>          sl_poutput(1, &sl_output[0]);
  d_slicer->work(sl_nout, sl_pinput, sl_poutput);
  if(d_disk_write) {
    sl_ofile.write((const char *)&sl_output[0], sl_output.size());
  }
  if(demod_debug) {
    std::cout << " SL ";
    fflush(stdout);
  }

  // Correlate
  long corr_nout = sl_output.size();
  std::vector<unsigned char>  corr_output(corr_nout);
  std::vector<const void*>    corr_pinput(1, &sl_output[0]);
  std::vector<void*>          corr_poutput(1, &corr_output[0]);
  d_corr->work(corr_nout, corr_pinput, corr_poutput);
  if(d_disk_write) {
    corr_ofile.write((const char *)&corr_output[0], corr_output.size());
  }
  if(demod_debug) {
    std::cout << " CORR ";
    fflush(stdout);
  }

  if(demod_debug)
    std::cout << " t_samples: " << t_samples << std::endl;

//  timestamp = timestamp - 64*(d_crq.size()+(c_tsamples-cf_nout));

  // Frame!
  pmt_dict_set(demod_properties, pmt_intern("timestamp"), pmt_from_long(timestamp));
  pmt_dict_set(demod_properties, pmt_intern("sps"), pmt_from_long(d_samples_per_symbol));
  pmt_dict_set(demod_properties, pmt_intern("bps"), pmt_from_long(BITS_PER_SYMBOL));
  // RSSI

  pmt_t p_corr_output = pmt_make_any(corr_output);
  d_cs->send(s_response_demod, pmt_list2(p_corr_output, demod_properties));

}

// The MAC layer connects to 'usrp_server' which has a control/status channel,
// a TX, and an RX port.  The MAC layer can then relay TX/RX data back and
// forth to the application, or a physical layer once available.
void gmsk::define_ports()
{
  // Ports applications used to connect to us
  d_cs = define_port("cs0", "gmsk-cs", true, mb_port::EXTERNAL);
}

// GR uses the numpy convolve function, heres a C++ replacement
void gmsk::convolve(float X[],float Y[], float Z[], int lenx, int leny)
{
  float *zptr,s,*xp,*yp;
  int lenz;
  int i,n,n_lo,n_hi;

  lenz=lenx+leny-1;

  zptr=Z;

  for (i=0;i<lenz;i++) {
    s=0.0;
    n_lo=0>(i-leny+1)?0:i-leny+1;
    n_hi=lenx-1<i?lenx-1:i;
    xp=X+n_lo;
    yp=Y+i-n_lo;
    for (n=n_lo;n<=n_hi;n++) {
      s+=*xp * *yp;
      xp++;
      yp--;
    }
  *zptr=s;
  zptr++;
  }
}

void gmsk::conv_to_binary(std::string code, std::vector<unsigned char> &output)
{
  for(int i=0; i < (int)code.length()/8; i++) {
    std::string curr = code.substr(i*8, 8);
    output[i] = (unsigned char)std::bitset<std::numeric_limits<unsigned char>::digits>(curr).to_ulong();
  }
}

REGISTER_MBLOCK_CLASS(gmsk);
