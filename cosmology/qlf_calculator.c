#include<stdio.h>
#include<stdlib.h>
#include<math.h>

double pei_dust_extinction(double lambda_in_microns);
double morrison_photoeletric_absorption(double x);
double cross_section(double nu);
double return_ratio_to_b_band(double nu);
double return_ratio_to_hard_xray(double nu);
double l_band(double log_l_bol, double nu);
double l_band_jacobian(double log_l_bol, double nu);
double l_band_dispersion(double log_l_bol, double nu);
double bol_lf_at_z(double log_l_bol, double z, int FIT_KEY);
double return_tau(double log_NH, double nu);
double distance_modulus(double z);



/* 
	Script to return the bolometric quasar luminosity function 
		and resulting luminosity functions in any band, from the compilation 
		of observations from 24microns to 30keV. 
		
	See Hopkins, Richards, & Hernquist, 2006, ApJ (HRH06)
	
	
	Compiled observations from Boyle et al. (2000), Cristiani et al. (2004), 
		Croom et al. (2004), Fan et al. (2001a,b,2003,2004), Hunt et al. (2004), 
		Kennefick et al. (1995), Richards et al. (2005, 2006a,b), Schmidt et al. (2006), 
		Siana et al. (2006), Wolf et al. (2003), Hasinger et al. (2005), 
		Miyaji et al. (2000, 2001), Silverman et al. (2005a,b,c), 
		Barger et al. (2003a,b,2005), Barger & Cowie (2005), La Franca et al. (2005), 
		Nandra et al. (2005), Sazonov & Revnivtsev (2004), Ueda et al. (2003), 
		Brown et al. (2006), Matute et al. (2006), Hao et al. (2005), 
		Elvis et al. (1994), Vanden Berk et al. (2001), Telfer et al. (2002), 
		Vignali et al. (2003), Hatziminaoglou et al. (2005), Shemmer et al. (2006), 
		Tozzi et al. (2006), Streteva et al. (2005), Steffen et al. (2006), 
		George et al. (1998), Perola et al. (2002), and others : 
			as detailed in HRH06


	number of input parameters : 2-3
		1 : nu -- the *rest frame* frequency of interest. for a specific band computed in 
			 HRH06, enter :
			 	0.0 = bolometric, -1.0 = B-band, -2.0 = mid-IR (15 microns)
			   -3.0 = soft X-ray (0.5-2 keV), -4.0 = hard X-ray (2-10 keV)
			otherwise, give nu in Hz, which is taken to be the effective 
			  frequency of the band (for which attenuation, etc. is calculated)
			(double)

		2 : redshift -- the redshift of the luminosity function to be returned
			(double)
		
		3 : fit_key -- optional flag which sets the fitted form of the bolometric QLF to 
			  adopt in calculating the observed QLF at nu. The fit parameters and 
			  descriptions are given in HRH06, 
			  specifically Tables 3 & 4
			  
			  0 = "Full" (default) the best-fit model derived therein 
			  		(double power law fit, allowing both  
			  	     the bright and faint-end slopes to evolve with redshift)
			  			chi^2/nu = 1007/508  (with respect to the compiled observations above)
			  1 = "PLE" 
			  		(pure luminosity evolution - double power-law with evolving Lstar but 
			  		 no change in the faint & bright end slopes)
			  			chi^2/nu = 1924/511 
			  2 = "Faint" 
			  		(double power-law with evolving Lstar and evolving faint-end slope,  
			  		 but fixed bright end slope)
			  			chi^2/nu = 1422/510 
			  3 = "Bright" 
			  		(double power-law with evolving Lstar and evolving bright-end slope,  
			  		 but fixed faint end slope)
			  			chi^2/nu = 1312/509 
			  4 = "Scatter" 
			  		(double power-law with evolution in both faint and bright end slopes, 
			  		 but adding a uniform ~0.05 dex to the error estimates sample-to-sample, 
			  		 to *roughly* compare systematic offsets)
			  			chi^2/nu = 445/508 (note, this fit has the error bars increased, and 
			  				also under-weights some of the most well-constrained samples, 
			  				so the reduced chi^2/nu should not be taken to mean it is 
			  				necessarily more accurate)
			  5 = "Schechter"
			  		(fitting a modified Schechter function, allowing both bright and 
			  		 faint end slopes to evolve with redshift, instead of a double power law)
			  			chi^2/nu = 1254/509 
			  6 = "LDDE"
			  		(luminosity-dependent density evolution fit)
			  			chi^2/nu = 1389/507
			  7 = "PDE"
			  		(pure density evolution fit)
			  		  	chi^2/nu = 3255/511
			  		  	
			  (the bolometric QLF is called from the 'bol_lf_at_z' routine : 
			  	an arbitrary bolometric QLF can be added there if desired)  	
			  		  	
	
	The results are simply printed to stdout, in five columns : 
		1 : observed luminosity in the band, in (log_{10}(L [erg/s]))
				(luminosities output are nu*L_nu, unless the bolometric, 
				 soft X-ray, or hard X-ray flags are set, in which case the luminosities 
				 are integrated over the appropriate frequency range)

		2 : corresponding absolute monochromatic AB magnitude at the given frequency nu. 
				(for bolometric, soft and hard X-ray bands which are integrated 
				 over some frequency range, this adopts effective frequencies of 
				 2500 Angstrom, 1 keV and 5 keV, respectively -- these are totally 
				 arbitrary, of course, but the AB magnitude is not well defined 
				 in any case for these examples). 

		3 : corresponding observed monochromatic flux in (log_{10}(S_nu [milliJanskys])) 
				(i.e. log_{10}( S_nu [10^-26 erg/s/cm^2/Hz] ))
				(This calculates the flux with the luminosity distance appropriate for the 
				 adopted cosmology in HRH06: Omega_Matter=0.3, Omega_Lambda=0.7, h=0.7. 
				 Attentuation from observed column density distributions is included, 
				 as described in the paper, but intergalactic attentuation appropriate 
				 near Lyman-alpha, for example, is NOT included in this calculation. 
				 Be careful, as this also does NOT include the K-correction, defined as : 
				      m = M + (distance modulus) + K
				      K = -2.5 * log[ (1+z) * L_{(1+z)*nu_obs} / L_{nu_obs} ]
				 For the bolometric, soft and hard X-rays, this returns the 
				 integrated flux over the 
				 appropriate frequency ranges, in CGS units (erg/s/cm^2).)

		4 : corresponding bolometric luminosity (given the median bolometric 
				corrections as a function of luminosity), in (log_{10}(L [erg/s]))

		5 : comoving number density per unit log_{10}(luminosity) : 
				dphi/dlog_{10}(L)  [ Mpc^{-3} log_{10}(L)^{-1} ]
				(make sure to correct by the appropriate factor to convert to e.g. 
				 the number density per unit magnitude) 


	Example use : 
        compile       ::    > gcc qlf_calculator.c -o qlf_calc

        standard call ::    > qlf_calc -1.0 2.0
             (print the B-band qlf at z=2 to stdout)

        dump to file  ::    > qlf_calc -1.0 2.0 > filename
             (the output results above get dumped to 'filename')

        compare different bolometric QLF models
                            > qlf_calc -1.0 2.0 1 > filename
             (as the above, but now fit_key = 1, so it uses the PLE bolometric QLF)



	An additional note -- this code is poorly adapted to handle the rest-frame wavelengths  
		near the Hydrogen absorption edge, (100 < lambda < 912 angstroms, 
		i.e. energies ~1-10 Rydberg), where the cross sections are 
		extremely high and a simple extinction curve application completely obscures that 
		portion of the spectrum. As such, this portion of the spectrum should be treated with 
		care -- if it is of interest, the "return_tau" routine can easily be modified to allow 
		some finite fixed or column-density dependent escape fraction. 

*/



void work(double nu, double REDSHIFT, int FIT_KEY, int n_l_bol_pts, const double * l_bol_grid, double * l_band_grid, double * M_AB_grid, double * S_nu_grid, double * phi_bol_grid) {
    // setup a grid of luminosities to consider
    double m_AB_obs, nu_eff;
    nu_eff = nu;
    if (nu ==  0.) {nu_eff = 1.1992000e15;} // 2500 angstrom
    if (nu == -1.) {nu_eff = 6.8136364e14;} // 4400 angstrom
    if (nu == -2.) {nu_eff = 1.9986667e13;} // 15 micron
    if (nu == -3.) {nu_eff = 2.4180000e17;} // 'effective' 1 keV
    if (nu == -4.) {nu_eff = 1.2090000e18;} // 'effective' 5 keV

    double *phi_grid_out = calloc(n_l_bol_pts,sizeof(double));
    int i_lbol,j_lbol;
    // first, load up the bolometric QLF
    for(i_lbol=0;i_lbol<n_l_bol_pts;i_lbol++) {
        phi_bol_grid[i_lbol] = bol_lf_at_z(l_bol_grid[i_lbol],REDSHIFT,FIT_KEY) 
            * l_band_jacobian(l_bol_grid[i_lbol],nu) ;
        l_band_grid[i_lbol] = log10(l_band(l_bol_grid[i_lbol],nu));
        phi_grid_out[i_lbol] = 0.0;
        M_AB_grid[i_lbol] = -2.5*l_band_grid[i_lbol]+2.5*log10(nu_eff)-32.38265724887536;
        // AB_nu for l_band_grid=nuLnu/L_sun_bol and nu in Hz
        m_AB_obs = M_AB_grid[i_lbol] - distance_modulus(REDSHIFT);
        S_nu_grid[i_lbol] = -0.4*(m_AB_obs-16.40); // returns log_{10}(S_nu/mJy)
        if ((nu == 0.) || (nu == -3.) || (nu == -4.)) {
            S_nu_grid[i_lbol] = l_band_grid[i_lbol] + 0.4*distance_modulus(REDSHIFT) 
                - 6.486937100449856; // last factor for CGS conversion, 
            //  given that l_band is in L_sun
        }
        S_nu_grid[i_lbol] = pow(10.,S_nu_grid[i_lbol]);
    }
    if (nu != 0.) {	// if just want bolometric QLF, we're done, otherwise, continue w. convolution

        // convolve over the dispersion in bolometric corrections, using the calculated 
        //    dispersions and assuming a lognormal distribution
        double lb0,sig0,prefac,expfac;
        for(i_lbol=0;i_lbol<n_l_bol_pts;i_lbol++) {
            lb0  = l_band_grid[i_lbol];
            sig0 = l_band_dispersion(l_bol_grid[i_lbol],nu);
            expfac = -0.5/(sig0*sig0);
            prefac = 1./(sig0 * sqrt(2.0*3.14159267)) * phi_bol_grid[i_lbol] * ((i_lbol<n_l_bol_pts-1)?(l_bol_grid[i_lbol+1] - l_bol_grid[i_lbol]):(l_bol_grid[i_lbol] - l_bol_grid[i_lbol-1]));
            for(j_lbol=0;j_lbol<n_l_bol_pts;j_lbol++) 
                phi_grid_out[j_lbol] += prefac * exp(expfac*(lb0-l_band_grid[j_lbol])*(lb0-l_band_grid[j_lbol])) ;
        }
        for(i_lbol=0;i_lbol<n_l_bol_pts;i_lbol++) {
            phi_bol_grid[i_lbol] = phi_grid_out[i_lbol];
            phi_grid_out[i_lbol] = 0.;
        }

        /* now, convolve over the distribution of column densities to get the 
        //   post-obscuration bolometric QLF, adopting the observed distribution from 
        //   Ueda et al. (2003)
        //
        //
        // this considers a flat NH function in each of several bins
        //		;; need to define a "lower limit" NH --> Ueda et al. use 20, our optical 
        //		;;   calculations seem to suggest that's about right, so fine
         */

        double NH_MIN  = 20.0;
        double NH_MAX  = 25.0;
        double D_NH    = 0.01;
        int    N_NH    = (int )((NH_MAX-NH_MIN)/D_NH + 1.0);
        int iNH;
        double *NH;
        NH = calloc(N_NH,sizeof(double));
        double *tau;
        tau = calloc(N_NH,sizeof(double));
        for(iNH=0;iNH<N_NH;iNH++) {
            NH[iNH] = NH_MIN + ((double )iNH)*D_NH;
            tau[iNH] = return_tau(NH[iNH],nu);
        }
        // loop over the LF and attenuate everything appropriately
        double eps,psi_44,beta_L,psi,psi_max,f_low,f_med,f_hig,f_compton,L_HX,NH_0,f_NH,dN_NH;
        double li,lo,p2,p1,p0;
        int n0 = 0;
        eps = 1.7;
        psi_44 = 0.47;
        beta_L = 0.10;
        psi_max = (1.+eps)/(3.+eps);
        double *l_obs_grid;
        l_obs_grid = calloc(n_l_bol_pts,sizeof(double));
        double *phi_obs_grid;
        phi_obs_grid = calloc(n_l_bol_pts,sizeof(double));
        for(iNH=0;iNH<N_NH;iNH++) {
            NH_0 = NH[iNH];		
            // need to interpolate to lay this over the grid already set up
            for(i_lbol=0;i_lbol<n_l_bol_pts;i_lbol++) {
                l_obs_grid[i_lbol] = l_band_grid[i_lbol] - tau[iNH]/log(10.);
                phi_obs_grid[i_lbol] = phi_bol_grid[i_lbol];
            }
            n0 = 0;
            for(i_lbol=0;i_lbol<n_l_bol_pts;i_lbol++) {
                li = l_band_grid[i_lbol];
                while((li >= l_obs_grid[n0+1]) && (n0+1 < n_l_bol_pts)) n0+=1;
                if (n0+1 < n_l_bol_pts) {
                    // interpolate between the two l_obs_grid points
                    p1 = log10(phi_obs_grid[n0]);
                    p2 = log10(phi_obs_grid[n0+1]);
                    p0 = p1 + (p2-p1) * ((li - l_obs_grid[n0])/(l_obs_grid[n0+1] - l_obs_grid[n0]));
                } else {
                    // extrapolate out
                    p1 = log10(phi_obs_grid[n_l_bol_pts-2]);
                    p2 = log10(phi_obs_grid[n_l_bol_pts-1]);
                    p0 = p1 + (p2-p1) * ((li - l_obs_grid[n_l_bol_pts-2])/(l_obs_grid[n_l_bol_pts-1] - l_obs_grid[n_l_bol_pts-2]));
                }
                L_HX  = log10(l_band(l_bol_grid[i_lbol],-4.0));
                //fprintf(stderr," lhx = %f \n",L_HX);
                psi = psi_44 - beta_L * (L_HX + log10(3.9) + 33.0 - 44.0);
                if (psi < 0.) psi = 0.;
                if (psi > psi_max) psi = psi_max;
                f_low = 2.0 - ((5.+2.*eps)/(1.+eps))*psi;
                f_med = (1./(1.+eps))*psi;
                f_hig = (eps/(1.+eps))*psi;
                f_compton = f_hig;
                f_low = f_low / (1. + f_compton);
                f_med = f_med / (1. + f_compton);
                f_hig = f_hig / (1. + f_compton);	
                f_NH = 0.0;
                if ((NH_0 <= 20.5)) f_NH = f_low;
                if ((NH_0 > 20.5) && (NH_0 <= 23.0)) f_NH = f_med;
                if ((NH_0 > 23.0) && (NH_0 <= 24.0)) f_NH = f_hig;
                if ((NH_0 > 24.0)) f_NH = f_compton;
                dN_NH = f_NH * D_NH;
                phi_grid_out[i_lbol] += pow(10.,p0) * dN_NH ;
            }
        }
        for(i_lbol=0;i_lbol<n_l_bol_pts;i_lbol++) {
            phi_bol_grid[i_lbol] = phi_grid_out[i_lbol];
            phi_grid_out[i_lbol] = 0.;
        }
        free(NH);
        free(tau);
        free(l_obs_grid);
        free(phi_obs_grid);
    }
    free(phi_grid_out);

    return;
}

int main(int argc, char** argv) 
{
	double nu,REDSHIFT;
	int FIT_KEY=0;
	int VOCAL=1;	// set to 0 to only print the output luminosity/QLF lists (i.e. no explanatory notes)

	// get the parameters from the call (could just as well make this a 
	//	 library script or completely self-contained, you 
	//   just need to pass or hardwire these three parameters 
	//   (nu, redshift, fit_key) 
	//   in the code or call
  	if (argc < 3) {
  	  fprintf(stderr, "Expected >=3 arguments (found %d)\n",argc); 
  	  exit(0); 
  	}
	nu       = atof(argv[1]);
	REDSHIFT = atof(argv[2]);
	if (argc >= 4) {FIT_KEY  = atoi(argv[3]);}	// if specified, use a different bolometric QLF fit

	if (VOCAL==1) {
		if (nu==0.0)  fprintf(stderr," nu = 0.0 (returning bolometric QLF) \n");
		if (nu==-1.0) fprintf(stderr," nu = -1.0 (returning B-band (440 nm) QLF) \n");
		if (nu==-2.0) fprintf(stderr," nu = -2.0 (returning mid-IR (15 micron) QLF) \n");
		if (nu==-3.0) fprintf(stderr," nu = -3.0 (returning soft X-ray (0.5-2 keV) QLF) \n");
		if (nu==-4.0) fprintf(stderr," nu = -4.0 (returning hard X-ray (2-10 keV) QLF) \n");
		if (nu > 0.0) fprintf(stderr," nu = %e   (returning QLF at effective frequency of %e Hz) \n",nu,nu);
	}
		if ((nu<0.0)&&(nu!=0.)&&(nu!=-1.)&&(nu!=-2.)&&(nu!=-3.)&&(nu!=-4.)) {
			fprintf(stderr," allowed frequency nu not entered : please give the frequency of the band in Hz, \n");
			fprintf(stderr,"   or enter 0.0 for bolometric, -1.0 for B-band, -2.0 for 15 micron, \n");
			fprintf(stderr,"   -3.0 for soft X-ray (0.5-2 keV), -4.0 for hard X-ray (2-10 keV) \n");
			exit(0);
		} 
	if (VOCAL==1) {
		if ((nu>3.2872807e+15)&&(nu<3.2872807e+16)) {
			fprintf(stderr," Warning :: \n");
			fprintf(stderr,"   this code is poorly adapted to handle the rest-frame wavelengths  \n");
			fprintf(stderr,"   near the Hydrogen absorption edge, (100 < lambda < 912 angstroms, \n");
			fprintf(stderr,"   i.e. energies ~1-10 Rydberg), where the cross sections are \n");
			fprintf(stderr,"   extremely high and a simple extinction curve application completely obscures that \n");
			fprintf(stderr,"   portion of the spectrum. As such, this portion of the spectrum should be treated with \n");
			fprintf(stderr,"   care -- if it is of interest, the 'return_tau' routine can easily be modified to allow \n");
			fprintf(stderr,"   some finite fixed or column-density dependent escape fraction. \n");
		}
		if (nu>7.25e+18) {
			fprintf(stderr," Warning :: \n");
			fprintf(stderr,"   the observations compiled do not extend to nu >~ 30 keV, so the spectral \n");
			fprintf(stderr,"   behavior here is being estimated with our 'best guess' reflection component model \n");
		}
		if ((nu<1.0e+13)&&(nu!=0.)&&(nu!=-1.)&&(nu!=-2.)&&(nu!=-3.)&&(nu!=-4.)) {
			fprintf(stderr," Warning :: \n");
			fprintf(stderr,"   the observations compiled do not extend to lambda >~ 30 microns, so the spectral \n");
			fprintf(stderr,"   behavior here is being estimated with our 'best guess' IR bump Rayleigh-Jeans truncation \n");
		}
	}
		if (REDSHIFT<0.) {
			fprintf(stderr," Please enter a redshift z > 0 \n");
			exit(0);
		}			
	if (VOCAL==1) {
		fprintf(stderr," redshift = %f \n",REDSHIFT);
		if (REDSHIFT>6.4) fprintf(stderr," Extrapolating to un-observed redshifts, take caution... \n");
	}		

		if ((FIT_KEY<0)||(FIT_KEY>7)) FIT_KEY=0;
	if (VOCAL==1) {
		if (FIT_KEY==0) {
			  fprintf(stderr," fit_key = 0 : 'Full' bolometric QLF fit \n");
			  fprintf(stderr,"                 (see Table 3 of HRH06; best-fit model) \n");
			  fprintf(stderr,"                  (allows both bright and faint-end slopes to evolve with redshift) \n");
			  fprintf(stderr,"                    chi^2/nu = 1007/508  (with respect to the compiled observations therein) \n");
		}			
		if (FIT_KEY==1) {
			  fprintf(stderr," fit_key = 1 : 'PLE' bolometric QLF fit \n");
			  fprintf(stderr,"                 (see Table 3 of HRH06; note this is *not* the best-fit model) \n");
			  fprintf(stderr,"                 (pure luminosity evolution - double power-law with evolving Lstar but  \n");
			  fprintf(stderr,"                   no change in the faint & bright end slopes) \n");
			  fprintf(stderr,"                    chi^2/nu = 1924/511  (with respect to the compiled observations therein) \n");
		}			
		if (FIT_KEY==2) {
			  fprintf(stderr," fit_key = 2 : 'Faint' bolometric QLF fit \n");
			  fprintf(stderr,"                 (see Table 3 of HRH06; note this is *not* the best-fit model) \n");
			  fprintf(stderr,"                 (double power-law with evolving Lstar and faint-end slope,  \n");
			  fprintf(stderr,"                   but no change in the bright end slope with redshift) \n");
			  fprintf(stderr,"                    chi^2/nu = 1422/510  (with respect to the compiled observations therein) \n");
		}			
		if (FIT_KEY==3) {
			  fprintf(stderr," fit_key = 3 : 'Bright' bolometric QLF fit \n");
			  fprintf(stderr,"                 (see Table 3 of HRH06; note this is *not* the best-fit model) \n");
			  fprintf(stderr,"                 (double power-law with evolving Lstar and bright-end slope,  \n");
			  fprintf(stderr,"                   but no change in the faint end slope with redshift) \n");
			  fprintf(stderr,"                    chi^2/nu = 1312/510  (with respect to the compiled observations therein) \n");
		}			
		if (FIT_KEY==4) {
			  fprintf(stderr," fit_key = 4 : 'Scatter' bolometric QLF fit \n");
			  fprintf(stderr,"                 (see Table 3 of HRH06; note this is *not* the best-fit model) \n");
			  fprintf(stderr,"                 (allows faint and bright-end slopes to evolve; adds uniform 0.05 dex to  \n");
			  fprintf(stderr,"                   measured errors as a means to estimate systematic offsets \n");
			  fprintf(stderr,"                    chi^2/nu = 445/508  (with respect to the compiled observations therein) \n");
			  fprintf(stderr,"                     (this chi^2/nu should not be taken seriously due to the process above, \n");
			  fprintf(stderr,"                        the fit should be considered primarily heuristic) \n");
		}			
		if (FIT_KEY==5) {
			  fprintf(stderr," fit_key = 5 : 'Schechter' bolometric QLF fit \n");
			  fprintf(stderr,"                 (see Table 3 of HRH06) \n");
			  fprintf(stderr,"                 (fitting a modified Schechter function bolometric QLF instead of  \n");
			  fprintf(stderr,"                  a double power law; allows Lstar and slopes to evolve with redshift ) \n");
			  fprintf(stderr,"                    chi^2/nu = 1254/509  (with respect to the compiled observations therein) \n");
		}			
		if (FIT_KEY==6) {
			  fprintf(stderr," fit_key = 6 : 'LDDE' bolometric QLF fit \n");
			  fprintf(stderr,"                 (see Table 4 of HRH06; note this is *not* the best-fit model) \n");
			  fprintf(stderr,"                 (luminosity-dependent density evolution (e.g. Schmidt & Green 1983) \n");
			  fprintf(stderr,"                    chi^2/nu = 1389/507  (with respect to the compiled observations therein) \n");
		}			
		if (FIT_KEY==7) {
			  fprintf(stderr," fit_key = 6 : 'PDE' bolometric QLF fit \n");
			  fprintf(stderr,"                 (see Table 4 of HRH06; note this is *not* the best-fit model) \n");
			  fprintf(stderr,"                 (pure density evolution \n");
			  fprintf(stderr,"                    chi^2/nu = 3255/511  (with respect to the compiled observations therein) \n");
		}			
	}

	double log_l_bol_min =  8.0;
	double log_l_bol_max = 18.0;
	double d_log_l_bol = 0.1;

    int n_l_bol_pts = (int )((log_l_bol_max-log_l_bol_min)/d_log_l_bol);
    double * l_bol_grid = calloc(n_l_bol_pts,sizeof(double));
    double * l_band_grid = calloc(n_l_bol_pts,sizeof(double));
    double * M_AB_grid = calloc(n_l_bol_pts,sizeof(double));
    double * S_nu_grid = calloc(n_l_bol_pts,sizeof(double));
    double * phi_bol_grid = calloc(n_l_bol_pts,sizeof(double));

    int i_lbol;

    for(i_lbol=0;i_lbol<n_l_bol_pts;i_lbol++) {
        l_bol_grid[i_lbol] = log_l_bol_min + ((double )i_lbol)*d_log_l_bol + 0.008935;
    }
    work(nu, REDSHIFT, FIT_KEY, n_l_bol_pts, l_bol_grid, l_band_grid, M_AB_grid, S_nu_grid, phi_bol_grid);

	if (VOCAL==1) {
	fprintf(stderr," Output : L_obs    M_AB_nu    S_nu     L_bol     phi \n");
	fprintf(stderr,"    L_obs   = observed L in band, in log_{10}(L/[erg/s]) \n");
	if ((nu==0.)||(nu==-3.)||(nu==-4.)) {
		fprintf(stderr,"               (integrated over the appropriate bandpass) \n");
	} else {
		fprintf(stderr,"               (output is nu*L_{nu} at the frequency above) \n");
	}	
	fprintf(stderr,"    M_AB_nu = monochromatic AB absolute magnitude \n");
	if ((nu==0.)) {
		fprintf(stderr,"               (effective wavelength chosen arbitrarily at 2500 Angstroms) \n");}
	if ((nu==-3.)) {
		fprintf(stderr,"               (effective wavelength chosen arbitrarily at 1 keV) \n");}
	if ((nu==-4.)) {
		fprintf(stderr,"               (effective wavelength chosen arbitrarily at 5 keV) \n");}
	if ((nu==0.)||(nu==-3.)||(nu==-4.)) {
	fprintf(stderr,"    S       = observed flux (integrated over the bandpass), \n");
	fprintf(stderr,"                in CGS units (erg/s/cm^2) \n");
	} else {
	fprintf(stderr,"    S_nu    = observed specific flux WITHOUT K-correction or bandpass \n");
	fprintf(stderr,"                redshifting; i.e.  just  L_nu/(4*pi*D_lum^2),  \n");
	fprintf(stderr,"                in mJy (10^{-26} erg/s/cm^2/Hz) \n");
	}
	fprintf(stderr,"    L_bol   = (roughly) corresponding bolometric L, from the \n");
	fprintf(stderr,"                median bolometric corrections adopted \n");
	fprintf(stderr,"    phi     = number density per unit log_{10}(L), in Mpc^{-3} \n");
	}

    double lsun=log10(3.9)+33.;
    for(i_lbol=0;i_lbol<n_l_bol_pts;i_lbol++) l_band_grid[i_lbol] += lsun;
    for(i_lbol=0;i_lbol<n_l_bol_pts;i_lbol++) l_bol_grid[i_lbol]  += lsun;
    //for(i_lbol=0;i_lbol<n_l_bol_pts;i_lbol++) fprintf(stderr," %f %f %e \n",l_band_grid[i_lbol],l_bol_grid[i_lbol],phi_bol_grid[i_lbol]);
    for(i_lbol=0;i_lbol<n_l_bol_pts;i_lbol++) printf(" %f %f %e %f %e \n",l_band_grid[i_lbol],M_AB_grid[i_lbol],S_nu_grid[i_lbol],l_bol_grid[i_lbol],phi_bol_grid[i_lbol]);


    free(l_bol_grid);
    free(l_band_grid);
    free(M_AB_grid);
    free(S_nu_grid);
    free(phi_bol_grid);
    return 0;
}


// returns the distance modulus for the cosmology adopted in HRH06 :: 
//   Omega_Matter = 0.3, Omega_Lambda = 0.7, h = 0.7
//   -- this function can easily be changed to return the appopriate distance 
//      modulus for any cosmology, but remember that the *observed* magnitudes 
//      are the unchanged quantities (i.e. changing this, one needs to change 
//      the luminosities and densities of the QLF accordingly)
//   -- for convenience, use a fitted function to the distance modulus for 
//      this cosmology, which is good to better than a factor of 0.0001 
//      at z<=10 and 0.001 at 10 < z < 30 (i.e. no significant source of error here)
double distance_modulus(double z)
{
double P0,P1,P2,P3,P4,P5,P6,P7,P8,P9,P10,P11,P12,P13,P14;
P0=-36.840151; P1=-13.167313; P2=0.39803874; 
P3=-0.33694804; P4=2.5504298; P5=0.56714463; 
P6=-0.50247992; P7=-0.044725351; P8=-0.80476843;
double x,d;
x = log10(1.+z);
d = P0 + P1*pow(x,P2) + P3*pow(x,P4) + P5*pow(x,P6) + P7*pow(x,P8);
return d;
}



// returns our fitted bolometric QLF phi(L,z) -- FIT_KEY tells the code which 
//   fit to use: 0=full model, 1=PLE, 2=Faint, 3=Bright, 4=Scatter, 5=Schechter, 6=LDDE, 7=PDE 
//   (names are those in Table 3-4 of Hopkins, Richards, & Hernquist, fits described therein)
double bol_lf_at_z(double log_l_bol, double z, int FIT_KEY)
{
double P0,P1,P2,P3,P4,P5,P6,P7,P8,P9,P10,P11,P12,P13,P14;
double xsi = log10((1. + z)/(1. + 2.));
double beta_min = 1.3;
	if (FIT_KEY==5) beta_min = 0.02;


// full model
if (FIT_KEY==0) {P0=-4.8250643; P1=13.035753;   P2=0.63150872; P3=-11.763560; P4=-14.249833; 
				 P5=0.41698725; P6=-0.62298947; P7=2.1744386;  P8=1.4599393;  P9=-0.79280099; 
				 P10=0.;P11=0.;P12=0.;P13=0.;P14=0.;}
// PLE
if (FIT_KEY==1) {P0=-4.733; P1=12.965; P2=0.749; P3=-8.03; P4=-4.40; P5=0.517; P6=0.; 
				 P7=2.096; P8=0.; P9=0.; P10=0.;P11=0.;P12=0.;P13=0.;P14=0.;}
// PLE + faint-end slope evolution
if (FIT_KEY==2) {P0=-4.630; P1=12.892; P2=0.717; P3=-8.10; P4=-3.90; P5=0.272; P6=-0.972; 
				 P7=2.048; P8=0.; P9=0.; P10=0.;P11=0.;P12=0.;P13=0.;P14=0.;}
// PLE + bright-end slope evolution
if (FIT_KEY==3) {P0=-4.930; P1=13.131; P2=0.360; P3=-11.63; P4=-10.68; P5=0.605; P6=0.; 
				 P7=2.350; P8=1.53; P9=-0.745; P10=0.;P11=0.;P12=0.;P13=0.;P14=0.;}
// full model with uniform scatter added (warning: this under-weights the best samples, so 
//   even though it has lower chi^2/nu, it's not necessarily a more accurate fit)
if (FIT_KEY==4) {P0=-4.815; P1=13.064; P2=0.356; P3=-11.69; P4=-9.18; P5=0.351; P6=-0.826; 
				 P7=2.359; P8=1.534; P9=-0.889; P10=0.;P11=0.;P12=0.;P13=0.;P14=0.;}
// modified schechter function fit
if (FIT_KEY==5) {P0=-3.579; P1=11.482; P2=-1.78; P3=-21.22; P4=-25.93; P5=0.013; P6=-3.760; 
				 P7=0.354; P8=1.794; P9=-0.784; P10=0.;P11=0.;P12=0.;P13=0.;P14=0.;}
// LDDE
if (FIT_KEY==6) {P0=-6.20; P1=45.99; P2=0.933; P3=2.20; P4=46.72; P5=1.852; P6=0.274; 
				 P7=5.95; P8=-1.65; P9=0.29; P10=-0.62; P11=0.;P12=0.;P13=0.;P14=0.;}
// PDE
if (FIT_KEY==7) {P0=-6.66; P1=46.64; P2=0.858; P3=2.09; P4=46.00; P5=1.852; P6=0.; 
				 P7=4.13; P8=-2.53; P9=0.;P10=0.;P11=0.;P12=0.;P13=0.;P14=0.;}

// double power-law forms
if (FIT_KEY <= 5) {
	double phi_star = P0;
		// phi_star is constant as a function of redshift
	double l_star   = P1 + P2*xsi + P3*xsi*xsi + P4*xsi*xsi*xsi;
		// l_star evolves (luminosity evolution) as a cubic polynomial in the 
		//    convenient variable xsi=log_{10}((1+z)/(1+z_ref)) with z_ref=2
	double gamma_1  = P5 * pow(10.,xsi*P6);
		// gamma_1 the faint-end slope -- optionally evolves with redshift
	double gamma_2  = 2.0 * P7 / (pow(10.,xsi*P8) + pow(10.,xsi*P9));
		// gamma_2 the bright-end slope -- optionally evolves with redshift
		if (gamma_2 < beta_min) gamma_2 = beta_min;
			// cap the bright-end evolution to prevent unphysical divergence
	double x = log_l_bol - l_star;
	if (FIT_KEY==5) {return pow(10.,(phi_star - gamma_1*x - pow(10.,x*gamma_2)/log(10.)));}
	return pow(10.,phi_star - log10(pow(10.,x*gamma_1) + pow(10.,x*gamma_2)));
}
// ldde form
if ((FIT_KEY > 5)&&(FIT_KEY <= 7)) {
	double x = log_l_bol + log10(3.9) + 33. - P1;
	double phi_0 = pow(10.,P0) / (pow(10.,x*P2) + pow(10.,x*P3));
	double p1 = P7 + P9  * (log_l_bol + log10(3.9)+33. - 46.);
	double p2 = P8 + P10 * (log_l_bol + log10(3.9)+33. - 46.);
	double xc = log_l_bol + log10(3.9) + 33. - P4;
	double zc = 0.; 
		if (xc <= 0.) zc = P5 * pow(10.,xc*P6);
		if (xc >  0.) zc = P5;
	double ed = 0.;
		if (z <= zc) ed = pow((1.+z),p1);
		if (z >  zc) ed = pow((1.+zc),(p1-p2)) * pow((1.+z),p2);
	return (phi_0 * ed);
}
return 0.;
}

// return the intrinsic band luminosity for some bolometric luminosity and frequency; 
//   nu = 0 (l_bol), -1 (B-band), -2 (15 microns), -3 (0.5-2 keV), -4 (2-10 keV), 
//   otherwise nu is the observed frequency and the return is nu*L_nu
double l_band(double log_l_bol, double nu)
{
	double x = log_l_bol - 10.;
	double lband = 0.;
	double P0,P1,P2,P3;
	if (nu==(0.))  return pow(10.,log_l_bol);
	if (nu < 0.) {
		if (nu==(-1.)) {P0=8.99833; P1=6.24800; P2=-0.370587; P3=-0.0115970;}
		if (nu==(-2.)) {P0=10.6615; P1=7.40282; P2=-0.370587; P3=-0.0115970;}
		if (nu==(-3.)) {P0=10.0287; P1=17.8653; P2=0.276804;  P3=-0.0199558;}
		if (nu==(-4.)) {P0=6.08087; P1=10.8331; P2=0.276802;  P3=-0.0199597;}
		lband = P0*pow(10.,P3*x) + P1*pow(10.,P2*x);
		return pow(10.,log_l_bol)/lband;
	}
	// if not one of the specified bands, then take advantage of the fact that our model 
	//   spectrum is not l-dependent below 500 angstroms or above 50 angstroms, so 
	//   just take the appropriate ratios to renormalize to those luminosities
	double nu_angstrom = (2.998e8)/(1.0e-10);
	double nu500 = nu_angstrom/500.;
	double nu50  = nu_angstrom/50.;
	if (nu <= nu500) {
		// just take the ratio relative to B-band
		double P[4] = {8.99833   ,   6.24800  ,  -0.370587  , -0.0115970};
		lband = P[0]*pow(10.,P[3]*x) + P[1]*pow(10.,P[2]*x);
		return return_ratio_to_b_band(nu)*pow(10.,log_l_bol)/lband;
	}
	if (nu >= nu50) {
		// just take the ratio relative to the hard X-rays
		double P[4] = {6.08087   ,   10.8331  ,   0.276802  , -0.0199597};
		lband = P[0]*pow(10.,P[3]*x) + P[1]*pow(10.,P[2]*x);
		return return_ratio_to_hard_xray(nu)*pow(10.,log_l_bol)/lband;
	}
	if ((nu>nu500)&&(nu<nu50)) {
		// interpolate between both regimes
		double P[4] = {8.99833   ,   6.24800  ,  -0.370587  , -0.0115970};
		double L500 = return_ratio_to_b_band(nu500)/(P[0]*pow(10.,P[3]*x)+P[1]*pow(10.,P[2]*x));
		double Q[4] = {6.08087   ,   10.8331  ,   0.276802  , -0.0199597};
		double L50  = return_ratio_to_hard_xray(nu50)/(Q[0]*pow(10.,Q[3]*x)+Q[1]*pow(10.,Q[2]*x));
		double L00  = log10(L500) + log10(L50/L500) * (log10(nu/nu500)/log10(nu50/nu500));	
		return pow(10.,L00)*pow(10.,log_l_bol);
	}
}
// return the appropriate jacobian factors for the above (dlogL/dlogL_band)
//   nu = 0 (l_bol), -1 (B-band), -2 (15 microns), -3 (0.5-2 keV), -4 (2-10 keV), 
//   otherwise nu is the observed frequency 
double l_band_jacobian(double log_l_bol, double nu)
{
	double x = log_l_bol - 10.;
	double lband = 0.;
	double P0,P1,P2,P3;
	double D1,D2;
	if (nu==(0.))  return 1.0;
	if (nu < 0.) {
		if (nu==(-1.)) {P0=8.99833; P1=6.24800; P2=-0.370587; P3=-0.0115970;}
		if (nu==(-2.)) {P0=10.6615; P1=7.40282; P2=-0.370587; P3=-0.0115970;}
		if (nu==(-3.)) {P0=10.0287; P1=17.8653; P2=0.276804;  P3=-0.0199558;}
		if (nu==(-4.)) {P0=6.08087; P1=10.8331; P2=0.276802;  P3=-0.0199597;}
			D1 = P0*(1.+P3)*pow(10.,P3*x) + P1*(1.+P2)*pow(10.,P2*x);
			D2 = P0*pow(10.,P3*x) + P1*pow(10.,P2*x);
		return D1/D2;
	}
	// if not one of the specified bands, then take advantage of the fact that our model 
	//   spectrum is not l-dependent below 500 angstroms or above 50 angstroms, so 
	//   just take the appropriate ratios to renormalize to those luminosities
	double nu_angstrom = (2.998e8)/(1.0e-10);
	double nu500 = nu_angstrom/500.;
	double nu50  = nu_angstrom/50.;
	if (nu <= nu500) {
		// just take the ratio relative to B-band
		double P[4] = {8.99833   ,   6.24800  ,  -0.370587  , -0.0115970};
			D1 = P[0]*(1.+P[3])*pow(10.,P[3]*x) + P[1]*(1.+P[2])*pow(10.,P[2]*x);
			D2 = P[0]*pow(10.,P[3]*x) + P[1]*pow(10.,P[2]*x);
		return D1/D2;
	}
	if (nu >= nu50) {
		// just take the ratio relative to the hard X-rays
		double P[4] = {6.08087   ,   10.8331  ,   0.276802  , -0.0199597};
			D1 = P[0]*(1.+P[3])*pow(10.,P[3]*x) + P[1]*(1.+P[2])*pow(10.,P[2]*x);
			D2 = P[0]*pow(10.,P[3]*x) + P[1]*pow(10.,P[2]*x);
		return D1/D2;
	}
	if ((nu>nu500)&&(nu<nu50)) {
		// interpolate between both regimes
		double P[4] = {8.99833   ,   6.24800  ,  -0.370587  , -0.0115970};
			D1 = P[0]*(1.+P[3])*pow(10.,P[3]*x) + P[1]*(1.+P[2])*pow(10.,P[2]*x);
			D2 = P[0]*pow(10.,P[3]*x) + P[1]*pow(10.,P[2]*x);
		double L500=D1/D2;
		double Q[4] = {6.08087   ,   10.8331  ,   0.276802  , -0.0199597};
			D1 = Q[0]*(1.+Q[3])*pow(10.,Q[3]*x) + Q[1]*(1.+Q[2])*pow(10.,Q[2]*x);
			D2 = Q[0]*pow(10.,Q[3]*x) + Q[1]*pow(10.,Q[2]*x);
		double L50=D1/D2;
		double L00  = log10(L500) + log10(L50/L500) * (log10(nu/nu500)/log10(nu50/nu500));	
		return pow(10.,L00);
	}
}
// return the lognormal dispersion in bolometric corrections for a given band and luminosity; 
//   nu = 0 (l_bol), -1 (B-band), -2 (15 microns), -3 (0.5-2 keV), -4 (2-10 keV), 
//   otherwise nu is the observed frequency 
double l_band_dispersion(double log_l_bol, double nu)
{
	double x = log_l_bol - 9.;
	double lband = 0.;
	double s0,s1,beta,sf1,sf2,sx,sx_floor;
	sx_floor = 0.050; // minimum value of dispersion to enforce
	if (nu==(0.))  return 0.01;
	if (nu < 0.) {
		if (nu==(-1.)) {s0 = 0.08; beta = -0.20; s1 = 0.065;}
		if (nu==(-2.)) {s0 = 0.03; beta = -0.10; s1 = 0.095;}
		if (nu==(-3.)) {s0 = 0.01; beta =  0.10; s1 = 0.060;}
		if (nu==(-4.)) {s0 = 0.04; beta =  0.05; s1 = 0.080;}
		return s0 * pow(10.,beta*x) + s1;
	}
	// interpolate between the known ranges, and (conservatively) hold constant 
	//   outside of them. roughly consistent with Richards et al. 2006 dispersions, 
	//   but uncertainty in how large the dispersions should be yields ~10% uncertainties 
	//   between 15microns and 10keV, and larger outside those ranges (since the 
	//   dispersions there are poorly determined) -- still, the lognormal dispersions 
	//   vary relatively weakly over observed ranges, so these are probably the 
	//   maximal uncertainties due to this effect
	double nu15 = 2.00e13;
	double nuBB = 6.81818e14;
	double nuSX = 0.5 * 2.418e17;
	double nuHX = 10. * 2.418e17;
	if (nu < nu15) {s0 = 0.03; beta = -0.10; s1 = 0.095; return s0*pow(10.,beta*x)+s1;}
	if (nu >=nuHX) {s0 = 0.04; beta =  0.05; s1 = 0.080; return s0*pow(10.,beta*x)+s1;}
	if ((nu >= nu15)&&(nu< nuBB)) {
		s0 = 0.03; beta = -0.10; s1 = 0.095;
			sf1 = s0 * pow(10.,beta*x) + s1;
		s0 = 0.08; beta = -0.20; s1 = 0.065;
			sf2 = s0 * pow(10.,beta*x) + s1;
		sx = sf1 + (sf2-sf1) * (log10(nu/nu15)/log10(nuBB/nu15));
		if (sx<=sx_floor) {sx=sx_floor;}
		return sx;
	}
	if ((nu >= nuBB)&&(nu< nuSX)) {
		s0 = 0.08; beta = -0.20; s1 = 0.065;
			sf1 = s0 * pow(10.,beta*x) + s1;
		s0 = 0.01; beta =  0.10; s1 = 0.060;
			sf2 = s0 * pow(10.,beta*x) + s1;
		sx = sf1 + (sf2-sf1) * (log10(nu/nu15)/log10(nuBB/nu15));
		if (sx<=sx_floor) {sx=sx_floor;}
		return sx;
	}
	if ((nu >= nuSX)&&(nu< nuHX)) {
		s0 = 0.01; beta =  0.10; s1 = 0.060;
			sf1 = s0 * pow(10.,beta*x) + s1;
		s0 = 0.04; beta =  0.05; s1 = 0.080;
			sf2 = s0 * pow(10.,beta*x) + s1;
		sx = sf1 + (sf2-sf1) * (log10(nu/nu15)/log10(nuBB/nu15));
		if (sx<=sx_floor) {sx=sx_floor;}
		return sx;
	}
}


// load the x-ray template, based on the observations in text and 
//     specifically the Magdziarz & Zdziarski 1995 PEXRAV model with Gamma=1.8 
//     (Tozzi et al., George et al.), theta=2pi, solar abundances
double return_ratio_to_hard_xray(double nu)
{
double log_nu[275]={
16.00, 16.02, 16.04, 16.06, 16.08, 16.10, 16.12, 16.14, 16.16, 16.18, 16.20, 16.22, 16.24,
16.26, 16.28, 16.30, 16.32, 16.34, 16.36, 16.38, 16.40, 16.42, 16.44, 16.46, 16.48, 16.50,
16.52, 16.54, 16.56, 16.58, 16.60, 16.62, 16.64, 16.66, 16.68, 16.70, 16.72, 16.74, 16.76,
16.78, 16.80, 16.82, 16.84, 16.86, 16.88, 16.90, 16.92, 16.94, 16.96, 16.98, 17.00, 17.02,
17.04, 17.06, 17.08, 17.10, 17.12, 17.14, 17.16, 17.18, 17.20, 17.22, 17.24, 17.26, 17.28,
17.30, 17.32, 17.34, 17.36, 17.38, 17.40, 17.42, 17.44, 17.46, 17.48, 17.50, 17.52, 17.54,
17.56, 17.58, 17.60, 17.62, 17.64, 17.66, 17.68, 17.70, 17.72, 17.74, 17.76, 17.78, 17.80,
17.82, 17.84, 17.86, 17.88, 17.90, 17.92, 17.94, 17.96, 17.98, 18.00, 18.02, 18.04, 18.06,
18.08, 18.10, 18.12, 18.14, 18.16, 18.18, 18.20, 18.22, 18.24, 18.26, 18.28, 18.30, 18.32,
18.34, 18.36, 18.38, 18.40, 18.42, 18.44, 18.46, 18.48, 18.50, 18.52, 18.54, 18.56, 18.58,
18.60, 18.62, 18.64, 18.66, 18.68, 18.70, 18.72, 18.74, 18.76, 18.78, 18.80, 18.82, 18.84,
18.86, 18.88, 18.90, 18.92, 18.94, 18.96, 18.98, 19.00, 19.02, 19.04, 19.06, 19.08, 19.10,
19.12, 19.14, 19.16, 19.18, 19.20, 19.22, 19.24, 19.26, 19.28, 19.30, 19.32, 19.34, 19.36,
19.38, 19.40, 19.42, 19.44, 19.46, 19.48, 19.50, 19.52, 19.54, 19.56, 19.58, 19.60, 19.62,
19.64, 19.66, 19.68, 19.70, 19.72, 19.74, 19.76, 19.78, 19.80, 19.82, 19.84, 19.86, 19.88,
19.90, 19.92, 19.94, 19.96, 19.98, 20.00, 20.02, 20.04, 20.06, 20.08, 20.10, 20.12, 20.14,
20.16, 20.18, 20.20, 20.22, 20.24, 20.26, 20.28, 20.30, 20.32, 20.34, 20.36, 20.38, 20.40,
20.42, 20.44, 20.46, 20.48, 20.50, 20.52, 20.54, 20.56, 20.58, 20.60, 20.62, 20.64, 20.66,
20.68, 20.70, 20.72, 20.74, 20.76, 20.78, 20.80, 20.82, 20.84, 20.86, 20.88, 20.90, 20.92,
20.94, 20.96, 20.98, 21.00, 21.02, 21.04, 21.06, 21.08, 21.10, 21.12, 21.14, 21.16, 21.18,
21.20, 21.22, 21.24, 21.26, 21.28, 21.30, 21.32, 21.34, 21.36, 21.38, 21.40, 21.42, 21.44,
21.46, 21.48 
};
double log_nuLnu[275]={
-2.1132, -2.1092, -2.1052, -2.1012, -2.0972, -2.0932, -2.0892, -2.0852, -2.0812, -2.0772, -2.0732, 
-2.0692, -2.0652, -2.0612, -2.0572, -2.0532, -2.0492, -2.0452, -2.0412, -2.0372, -2.0332, -2.0292, 
-2.0252, -2.0212, -2.0172, -2.0132, -2.0092, -2.0052, -2.0012, -1.9972, -1.9932, -1.9892, -1.9852, 
-1.9812, -1.9772, -1.9732, -1.9692, -1.9652, -1.9611, -1.9571, -1.9531, -1.9491, -1.9452, -1.9412, 
-1.9372, -1.9332, -1.9292, -1.9252, -1.9212, -1.9172, -1.9132, -1.9092, -1.9052, -1.9012, -1.8971, 
-1.8931, -1.8894, -1.8854, -1.8814, -1.8774, -1.8734, -1.8694, -1.8654, -1.8614, -1.8574, -1.8534, 
-1.8495, -1.8455, -1.8415, -1.8374, -1.8334, -1.8294, -1.8253, -1.8213, -1.8172, -1.8132, -1.8091, 
-1.8050, -1.8009, -1.7968, -1.7927, -1.7885, -1.7843, -1.7803, -1.7761, -1.7718, -1.7675, -1.7631, 
-1.7587, -1.7547, -1.7502, -1.7457, -1.7410, -1.7363, -1.7314, -1.7266, -1.7215, -1.7163, -1.7109, 
-1.7053, -1.6999, -1.6940, -1.6878, -1.6814, -1.6747, -1.6677, -1.6604, -1.6527, -1.6451, -1.6369, 
-1.6283, -1.6194, -1.6342, -1.6266, -1.6188, -1.6113, -1.6036, -1.5950, -1.5861, -1.5763, -1.5655, 
-1.5530, -1.5391, -1.5253, -1.5116, -1.4968, -1.4808, -1.4635, -1.4451, -1.4259, -1.4062, -1.3867, 
-1.3676, -1.3494, -1.3323, -1.3165, -1.3021, -1.2891, -1.2775, -1.2671, -1.2580, -1.2501, -1.2432, 
-1.2372, -1.2320, -1.2277, -1.2240, -1.2210, -1.2185, -1.2166, -1.2152, -1.2143, -1.2141, -1.2144, 
-1.2150, -1.2160, -1.2172, -1.2188, -1.2207, -1.2229, -1.2253, -1.2280, -1.2310, -1.2342, -1.2376, 
-1.2413, -1.2453, -1.2495, -1.2540, -1.2588, -1.2640, -1.2694, -1.2752, -1.2814, -1.2881, -1.2952, 
-1.3028, -1.3109, -1.3196, -1.3289, -1.3389, -1.3496, -1.3609, -1.3731, -1.3861, -1.4000, -1.4148, 
-1.4302, -1.4456, -1.4602, -1.4741, -1.4877, -1.5012, -1.5147, -1.5284, -1.5424, -1.5567, -1.5714, 
-1.5866, -1.6023, -1.6186, -1.6354, -1.6529, -1.6711, -1.6901, -1.7098, -1.7304, -1.7518, -1.7742, 
-1.7975, -1.8219, -1.8474, -1.8740, -1.9018, -1.9308, -1.9612, -1.9930, -2.0263, -2.0611, -2.0975, 
-2.1356, -2.1755, -2.2173, -2.2611, -2.3070, -2.3551, -2.4055, -2.4584, -2.5138, -2.5720, -2.6331, 
-2.6971, -2.7642, -2.8347, -2.9087, -2.9864, -3.0678, -3.1533, -3.2430, -3.3371, -3.4358, -3.5394, 
-3.6481, -3.7620, -3.8814, -4.0067, -4.1382, -4.2760, -4.4205, -4.5719, -4.7306, -4.8971, -5.0716, 
-5.2545, -5.4462, -5.6471, -5.8576, -6.0783, -6.3096, -6.5520, -6.8059, -7.0719, -7.3506, -7.6428, 
-7.9489, -8.2696, -8.6055, -8.9574, -9.3260, -9.7124, -10.1171, -10.5410, 
-10.9850, -11.4499, -11.9371};

// want the ratio with respect to the intrinsic 2-10 keV:
double L_HX  = -1.47406;
double log_nu_obs = log10(nu);
double nuLnu_obs = 0.;

if (log_nu_obs < log_nu[0])   nuLnu_obs = log_nuLnu[0];
if (log_nu_obs > log_nu[273]) nuLnu_obs = -2.0 - pow(10.0,log_nu_obs-20.1204)*(0.43429448);
if ((log_nu_obs>=log_nu[0])&&(log_nu_obs<=log_nu[273])) {
	int n0 = (int )((log_nu_obs-log_nu[0])/0.02);
	nuLnu_obs = log_nuLnu[n0] + (log_nuLnu[n0+1]-log_nuLnu[n0]) * 
									((log_nu_obs-log_nu[n0])/(log_nu[n0+1]-log_nu[n0]));
}
return pow(10.0,nuLnu_obs-L_HX);
}



// load the optical-IR template, based on the observations in text and specifically 
//		 the Richards et al. 2006 mean blue SED 
double return_ratio_to_b_band(double nu)
{
double log_nu[226]={
12.50, 12.52, 12.54, 12.56, 12.58, 12.60, 12.62, 12.64, 12.66, 12.68, 12.70, 12.72, 12.74,
12.76, 12.78, 12.80, 12.82, 12.84, 12.86, 12.88, 12.90, 12.92, 12.94, 12.96, 12.98, 13.00,
13.02, 13.04, 13.06, 13.08, 13.10, 13.12, 13.14, 13.16, 13.18, 13.20, 13.22, 13.24, 13.26,
13.28, 13.30, 13.32, 13.34, 13.36, 13.38, 13.40, 13.42, 13.44, 13.46, 13.48, 13.50, 13.52,
13.54, 13.56, 13.58, 13.60, 13.62, 13.64, 13.66, 13.68, 13.70, 13.72, 13.74, 13.76, 13.78,
13.80, 13.82, 13.84, 13.86, 13.88, 13.90, 13.92, 13.94, 13.96, 13.98, 14.00, 14.02, 14.04,
14.06, 14.08, 14.10, 14.12, 14.14, 14.16, 14.18, 14.20, 14.22, 14.24, 14.26, 14.28, 14.30,
14.32, 14.34, 14.36, 14.38, 14.40, 14.42, 14.44, 14.46, 14.48, 14.50, 14.52, 14.54, 14.56,
14.58, 14.60, 14.62, 14.64, 14.66, 14.68, 14.70, 14.72, 14.74, 14.76, 14.78, 14.80, 14.82,
14.84, 14.86, 14.88, 14.90, 14.92, 14.94, 14.96, 14.98, 15.00, 15.02, 15.04, 15.06, 15.08,
15.10, 15.12, 15.14, 15.16, 15.18, 15.20, 15.22, 15.24, 15.26, 15.28, 15.30, 15.32, 15.34,
15.36, 15.38, 15.40, 15.42, 15.44, 15.46, 15.48, 15.50, 15.52, 15.54, 15.56, 15.58, 15.60,
15.62, 15.64, 15.66, 15.68, 15.70, 15.72, 15.74, 15.76, 15.78, 15.80, 15.82, 15.84, 15.86,
15.88, 15.90, 15.92, 15.94, 15.96, 15.98, 16.00, 16.02, 16.04, 16.06, 16.08, 16.10, 16.12,
16.14, 16.16, 16.18, 16.20, 16.22, 16.24, 16.26, 16.28, 16.30, 16.32, 16.34, 16.36, 16.38,
16.40, 16.42, 16.44, 16.46, 16.48, 16.50, 16.52, 16.54, 16.56, 16.58, 16.60, 16.62, 16.64,
16.66, 16.68, 16.70, 16.72, 16.74, 16.76, 16.78, 16.80, 16.82, 16.84, 16.86, 16.88, 16.90,
16.92, 16.94, 16.96, 16.98, 17.00
};
double log_nuLnu[226]={
44.39, 44.44, 44.50, 44.55, 44.60, 44.65, 44.70, 44.74, 44.78, 44.82, 44.86, 44.89, 44.92,
44.95, 44.97, 45.00, 45.02, 45.04, 45.06, 45.08, 45.10, 45.12, 45.14, 45.16, 45.17, 45.19,
45.20, 45.22, 45.23, 45.24, 45.25, 45.26, 45.27, 45.28, 45.29, 45.30, 45.31, 45.31, 45.32,
45.33, 45.34, 45.34, 45.35, 45.35, 45.36, 45.36, 45.37, 45.38, 45.38, 45.38, 45.39, 45.39,
45.39, 45.40, 45.40, 45.40, 45.40, 45.41, 45.41, 45.41, 45.41, 45.41, 45.41, 45.41, 45.41,
45.41, 45.40, 45.40, 45.40, 45.40, 45.40, 45.40, 45.39, 45.39, 45.38, 45.38, 45.37, 45.36,
45.35, 45.34, 45.33, 45.31, 45.30, 45.28, 45.26, 45.24, 45.22, 45.19, 45.17, 45.15, 45.13,
45.12, 45.11, 45.11, 45.11, 45.11, 45.12, 45.12, 45.13, 45.14, 45.15, 45.16, 45.18, 45.19,
45.20, 45.22, 45.23, 45.25, 45.26, 45.27, 45.28, 45.30, 45.32, 45.34, 45.36, 45.38, 45.40,
45.42, 45.44, 45.47, 45.49, 45.52, 45.55, 45.58, 45.60, 45.62, 45.64, 45.65, 45.65, 45.66,
45.66, 45.67, 45.68, 45.69, 45.71, 45.72, 45.74, 45.75, 45.77, 45.78, 45.79, 45.80, 45.80,
45.80, 45.80, 45.79, 45.77, 45.74, 45.71, 45.68, 45.64, 45.60, 45.57, 45.55, 45.52, 45.51,
45.49, 45.47, 45.46, 45.44, 45.42, 45.40, 45.38, 45.36, 45.34, 45.32, 45.29, 45.27, 45.25,
45.22, 45.20, 45.18, 45.15, 45.13, 45.11, 45.08, 45.06, 45.04, 45.02, 44.99, 44.97, 44.95,
44.92, 44.90, 44.88, 44.86, 44.83, 44.81, 44.79, 44.77, 44.74, 44.72, 44.70, 44.67, 44.65,
44.63, 44.61, 44.58, 44.56, 44.54, 44.52, 44.50, 44.47, 44.45, 44.43, 44.41, 44.40, 44.38,
44.36, 44.35, 44.33, 44.32, 44.30, 44.29, 44.28, 44.27, 44.27, 44.26, 44.26, 44.26, 44.26,
44.26, 44.25, 44.25, 44.25, 44.25
};

// want the ratio with respect to the intrinsic B-band:
double nu_BB = 14.833657;
double L_BB  = 45.413656;
double log_nu_obs = log10(nu);
double nuLnu_obs = 0.;

if (log_nu_obs < log_nu[0])   nuLnu_obs = log_nuLnu[0] + 2.0*(log_nu_obs - log_nu[0]);
if (log_nu_obs > log_nu[224]) nuLnu_obs = log_nuLnu[224];
	// assumes Gamma=2.0; the calling code will actually use X-ray template for this case
if ((log_nu_obs>=log_nu[0])&&(log_nu_obs<=log_nu[224])) {
	int n0 = (int )((log_nu_obs-log_nu[0])/0.02);
	nuLnu_obs = log_nuLnu[n0] + (log_nuLnu[n0+1]-log_nuLnu[n0]) * 
									((log_nu_obs-log_nu[n0])/(log_nu[n0+1]-log_nu[n0]));
}
return pow(10.0,nuLnu_obs-L_BB);
}


// returns the attenuation / optical depth tau for a given NH and frequency
//   nu = 0 (l_bol), -1 (B-band), -2 (15 microns), -3 (0.5-2 keV), -4 (2-10 keV), 
//    -- this accounts for the fact that the SX and HX are full bands to be integrated over 
//        (in e.g. B-band the effects of this are negligible, but in the soft & hard X-ray the 
//			tau(NH) curve is not nearly as fast a transition as it would be at a single 
//          frequency). 
//	 otherwise nu is the observed frequency 
//
double return_tau(double log_NH, double nu)
{
	double c_light = 2.998e8;
	double tau_f;
	if (nu <= 0.) {
		if (nu== 0.) return 0.;	// no bolometric attenuation
		if (nu==-1.) return pow(10.,log_NH)*cross_section(c_light/(4400.0e-10)); // call at nu_B
		if (nu==-2.) return pow(10.,log_NH)*cross_section(c_light/(15.0e-6));	 // call at 15microns

if (nu==-3.) {
double NH[101] = {
16.00, 16.10, 16.20, 16.30, 16.40, 16.50, 16.60, 16.70, 16.80, 16.90, 17.00, 17.10, 17.20,
17.30, 17.40, 17.50, 17.60, 17.70, 17.80, 17.90, 18.00, 18.10, 18.20, 18.30, 18.40, 18.50,
18.60, 18.70, 18.80, 18.90, 19.00, 19.10, 19.20, 19.30, 19.40, 19.50, 19.60, 19.70, 19.80,
19.90, 20.00, 20.10, 20.20, 20.30, 20.40, 20.50, 20.60, 20.70, 20.80, 20.90, 21.00, 21.10,
21.20, 21.30, 21.40, 21.50, 21.60, 21.70, 21.80, 21.90, 22.00, 22.10, 22.20, 22.30, 22.40,
22.50, 22.60, 22.70, 22.80, 22.90, 23.00, 23.10, 23.20, 23.30, 23.40, 23.50, 23.60, 23.70,
23.80, 23.90, 24.00, 24.10, 24.20, 24.30, 24.40, 24.50, 24.60, 24.70, 24.80, 24.90, 25.00,
25.10, 25.20, 25.30, 25.40, 25.50, 25.60, 25.70, 25.80, 25.90, 26.00    
};
double tau[101] = {		
-0.00000124, -0.00000158,    -0.00000197,    -0.00000249,    -0.00000313,    -0.00000393,   
-0.00000494, -0.00000624,    -0.00000784,    -0.00000989,    -0.00001245,    -0.00001566,
-0.00001973,    -0.00002483,    -0.00003127,    -0.00003935,    -0.00004955,    -0.00006236,
-0.00007852,    -0.00009884,    -0.00012443,    -0.00015666,    -0.00019722, -0.00024827,   
-0.00031253,    -0.00039344,    -0.00049527,    -0.00062347,    -0.00078482, -0.00098790,   
-0.00124348,    -0.00156514,    -0.00196988,    -0.00247909, -0.00311972,    -0.00392549,   
-0.00493867,    -0.00621233,    -0.00781276,    -0.00982293, -0.01234610,    -0.01551090,   
-0.01947660,    -0.02443970,    -0.03064190, -0.03837750,    -0.04800200,    -0.05994050,   
-0.07469200,    -0.09283250,    -0.11500600, -0.14191000,    -0.17425400,    -0.21271700,   
-0.25787899,    -0.31016999, -0.36986199,    -0.43716601,    -0.51242900,    -0.59641999,   
-0.69052601,    -0.79681402, -0.91799599,    -1.05745995,    -1.21945000,    -1.40935004,   
-1.63397002, -1.90190005,    -2.22390008,    -2.61347008,    -3.08738995,    -3.66647005,   
-4.37638998, -5.24916983,    -6.32518005,    -7.65556002,    -9.30533981,   -11.35690022,
-13.40830040,   -15.45989990,   -17.51140022,   -19.56290054,   -21.61440086, -23.66589928, 
 -25.71750069,   -27.76889992,   -29.82049942,   -31.87199974, -33.92350006,   -35.97499847,
  -38.02650070,   -40.07799911,   -42.12919998, -44.15449905,   -46.17983627,  
-48.20517349,   -50.23051071,   -52.25585175, -54.28115082,   -56.30648804,   -58.33182526
};
if (log_NH < NH[0])  tau_f = tau[0]  + (tau[1]-tau[0])   *(log_NH-NH[0])/(NH[1]-NH[0]);
if (log_NH > NH[99]) tau_f = tau[99] + (tau[100]-tau[99])*(log_NH-NH[99])/(NH[100]-NH[99]);
if ((log_NH>=NH[0])&&(log_NH<=NH[99])) {
	int n0 = (int )((log_NH-NH[0])/0.10);
	tau_f  = tau[n0] + (tau[n0+1]-tau[n0])*(log_NH-NH[n0])/(NH[n0+1]-NH[n0]);
}
if (tau_f >= 0.) tau_f=0.;
return -tau_f * log(10.);
}

if (nu==-4.) {
double NH[101] = {
16.00, 16.10, 16.20, 16.30, 16.40, 16.50, 16.60, 16.70, 16.80, 16.90, 17.00, 17.10, 17.20,
17.30, 17.40, 17.50, 17.60, 17.70, 17.80, 17.90, 18.00, 18.10, 18.20, 18.30, 18.40, 18.50,
18.60, 18.70, 18.80, 18.90, 19.00, 19.10, 19.20, 19.30, 19.40, 19.50, 19.60, 19.70, 19.80,
19.90, 20.00, 20.10, 20.20, 20.30, 20.40, 20.50, 20.60, 20.70, 20.80, 20.90, 21.00, 21.10,
21.20, 21.30, 21.40, 21.50, 21.60, 21.70, 21.80, 21.90, 22.00, 22.10, 22.20, 22.30, 22.40,
22.50, 22.60, 22.70, 22.80, 22.90, 23.00, 23.10, 23.20, 23.30, 23.40, 23.50, 23.60, 23.70,
23.80, 23.90, 24.00, 24.10, 24.20, 24.30, 24.40, 24.50, 24.60, 24.70, 24.80, 24.90, 25.00,
25.10, 25.20, 25.30, 25.40, 25.50, 25.60, 25.70, 25.80, 25.90, 26.00    
};
double tau[101] = {		
-0.00000005, -0.00000005,    -0.00000005,    -0.00000008,    -0.00000010,    -0.00000013,   
-0.00000016, -0.00000021,    -0.00000026,    -0.00000031,    -0.00000041,    -0.00000052,
-0.00000065,    -0.00000080,    -0.00000101,    -0.00000127,    -0.00000160,    -0.00000202,
-0.00000256,    -0.00000321,    -0.00000404,    -0.00000510,    -0.00000642, -0.00000808,   
-0.00001017,    -0.00001281,    -0.00001613,    -0.00002030,    -0.00002555, -0.00003218,   
-0.00004051,    -0.00005100,    -0.00006420,    -0.00008082, -0.00010174,    -0.00012808,   
-0.00016122,    -0.00020297,    -0.00025549,    -0.00032160, -0.00040484,    -0.00050955,   
-0.00064138,    -0.00080723,    -0.00101594, -0.00127847,    -0.00160871,    -0.00202394,   
-0.00254596,    -0.00320199,    -0.00402598, -0.00506041,    -0.00635801,    -0.00798424,   
-0.01002010,    -0.01256490, -0.01574020,    -0.01969330,    -0.02460080,    -0.03067190,   
-0.03814980,    -0.04731170, -0.05846370,    -0.07193180,    -0.08804700,    -0.10712300,   
-0.12943600, -0.15520699,    -0.18461201,    -0.21782200,    -0.25508299,    -0.29682499,   
-0.34375799, -0.39693701,    -0.45780700,    -0.52823699,    -0.61060297,    -0.70790398,
-0.82392102,    -0.96342301,    -1.13239002,    -1.33829999,    -1.59043002,    -1.90021002,
-2.28153992,    -2.75111008,    -3.32871008,    -4.03739023,    -4.90411997, -5.96181011,   
-7.25338984,    -8.54498959,    -9.83658981,   -11.12819958,   -12.41979980, -13.71140003,  
-15.00300026,   -16.29459953,   -17.58620071,   -18.87779999, -20.16939926
};
if (log_NH < NH[0])  tau_f = tau[0]  + (tau[1]-tau[0])   *(log_NH-NH[0])/(NH[1]-NH[0]);
if (log_NH > NH[99]) tau_f = tau[99] + (tau[100]-tau[99])*(log_NH-NH[99])/(NH[100]-NH[99]);
if ((log_NH>=NH[0])&&(log_NH<=NH[99])) {
	int n0 = (int )((log_NH-NH[0])/0.10);
	tau_f  = tau[n0] + (tau[n0+1]-tau[n0])*(log_NH-NH[n0])/(NH[n0+1]-NH[n0]);
}
if (tau_f >= 0.) tau_f=0.;
return -tau_f * log(10.);
}
}
return pow(10.,log_NH) * cross_section(nu);
}


// returns the cross section for absorption for a given nu in Hz
double cross_section(double nu)
{
	double sigma = 0.;
	double metallicity_over_solar = 1.;
	double keV_in_Hz = 2.418e17;
	double c_light = 2.998e8;
	double micron  = 1.0e-6;

/*
  ; For optical-IR regions, we use the Pei numerical approximations below.
  ;
  ; xsi = tau(lambda)/tau(B) is the ratio of extinction at lambda to the 
  ;    extinction in the B-band. 
  ; k = 10^21 (tau_B / NH)   (NH in cm^2) gives the dimensionless gas-to-dust
  ;    ratio, with k=0.78 for MW, k=0.16 for LMC, k=0.08 for SMC.
  ;    k is INDEPENDENT of the grain properties, and seems to scale rougly
  ;    linearly with metallicity
  ; so, for now, assume solar metallicity and k = k_MW = 0.78. we can rescale later.
  ;
  ; tau_B = ( NH / (10^21 cm^-2) ) * k --> SIGMA_B = k*10^-21  cm^2
  ; tau_lambda = xsi * tau_B --> SIGMA = xsi * SIGMA_B
  ;
  ; k = 0.78 for the MW
  ; k = 0.08 for the SMC, approximately in line with the MW/LMC/SMC metallicity 
  ;  sequence, so we take a k_MW then scaled by the metallicity
*/
	double k_dust_to_gas = 0.78 * metallicity_over_solar;
	double lambda_microns = c_light / nu / micron;
	if (nu < 0.03*keV_in_Hz) 
		sigma += pei_dust_extinction(lambda_microns) * k_dust_to_gas * 1.0e-21;


/*
  ; For 0.03 keV < E < 10 keV  
  ;   (7.2e15 < nu[Hz] < 2.4e18  or   1.2 < lambda[Angstroms] < 413)
  ;   we use the photoelectric absorption cross sections of 
  ;   Morrison & McCammon (1983)
  ;     NOTE: these assume solar abundances and no ionization, 
  ;             the appropriate number probably scales linearly with both
  ;   (this is all for the COMPTON THIN regime)
*/
	if (nu > (0.03*keV_in_Hz*1.362/3.0)) // above Lyman edge
		sigma += morrison_photoeletric_absorption(nu/keV_in_Hz);


/*
  ; Floor in cross-section set by non-relativistic (achromatic) Thompson scattering
  ;  (technically want to calculate self-consistently for the ionization state of the 
  ;   gas, but for reasonable values this agrees well with more detailed calculations 
  ;   including line effects from Matt, Pompilio, & La Franca; since we don't know the 
  ;   state of the gas (& have already calculated the inner reflection component), this
  ;   is the best guess)
*/
	sigma += 6.65e-25;;

return sigma;
}



double pei_dust_extinction(double lambda_in_microns)
{
int MW_key=0;
int LMC_key=0;
int SMC_key=1;
int i;
double xsi = 0.0*lambda_in_microns;
if (MW_key==1) {
  double a[6] = {165., 14., 0.045, 0.002, 0.002, 0.012};
  double l[6] = {0.047, 0.08, 0.22, 9.7, 18., 25.};
  double b[6] = {90., 4.00, -1.95, -1.95, -1.80, 0.00};
  double n[6] = {2.0, 6.5, 2.0, 2.0, 2.0, 2.0};
  double R_V = 3.08;
  for(i=0;i<6;i++) xsi += a[i] / ( pow(lambda_in_microns/l[i],n[i]) + pow(l[i]/lambda_in_microns,n[i]) + b[i] );
}
if (LMC_key==1) {
  double a[6] = {175., 19., 0.023, 0.005, 0.006, 0.020};
  double l[6] = {0.046, 0.08, 0.22, 9.7, 18., 25.};
  double b[6] = {90., 5.50, -1.95, -1.95, -1.80, 0.00};
  double n[6] = {2.0, 4.5, 2.0, 2.0, 2.0, 2.0};
  double R_V = 3.16;
  for(i=0;i<6;i++) xsi += a[i] / ( pow(lambda_in_microns/l[i],n[i]) + pow(l[i]/lambda_in_microns,n[i]) + b[i] );
}
if (SMC_key==1) {
  double a[6] = {185., 27., 0.005, 0.010, 0.012, 0.030};
  double l[6] = {0.042, 0.08, 0.22, 9.7, 18., 25.};
  double b[6] = {90., 5.50, -1.95, -1.95, -1.80, 0.00};
  double n[6] = {2.0, 4.0, 2.0, 2.0, 2.0, 2.0};
  double R_V = 2.93;
  for(i=0;i<6;i++) xsi += a[i] / ( pow(lambda_in_microns/l[i],n[i]) + pow(l[i]/lambda_in_microns,n[i]) + b[i] );
}
//double R_lam = (1.0 + R_V) * xsi;
return xsi;
}



double morrison_photoeletric_absorption(double x)	// x is nu in keV
{    
//  ; set the appropriate polynomial terms from Table 2 
//  ;   of Morrison & McCammon 1983 (for a given frequency range)
	double c0, c1, c2;
	if(x<0.03)
	{
		c0 =    17.3;
		c1 =   608.1;
		c2 = -2150.0;
		return (1.0e-24)*(c0+c1*0.03+c2*0.03*0.03)/(0.03*0.03*0.03)*pow(x/0.03,-2.43);
	}
	else if((x>=0.03)&&(x<0.1))
	{
		c0 =    17.3;
		c1 =   608.1;
		c2 = -2150.0;

	}else if((x>=0.1)&&(x<0.284))
	{
		c0 =    34.6;
		c1 =   267.9;
		c2 =  -476.1;

	}else if((x>=0.284)&&(x<0.4))
	{
		c0 =    78.1;
		c1 =    18.8;
		c2 =     4.3;

	}else if((x>=0.4)&&(x<0.532))
	{
		c0 =    71.4;
		c1 =    66.8;
		c2 =   -51.4;

	}else if((x>=0.532)&&(x<0.707))
	{
		c0 =    95.5;
		c1 =   145.8;
		c2 =   -61.1;

	}else if((x>=0.707)&&(x<0.867))
	{
		c0 =   308.9;
		c1 =  -380.6;
		c2 =   294.0;

	}else if((x>=0.867)&&(x<1.303))
	{
		c0 =   120.6;
		c1 =   169.3;
		c2 =   -47.7;

	}else if((x>=1.303)&&(x<1.840))
	{
		c0 =   141.3;
		c1 =   146.8;
		c2 =   -31.5;

	}else if((x>=1.840)&&(x<2.471))
	{
		c0 =   202.7;
		c1 =   104.7;
		c2 =   -17.0;

	}else if((x>=2.471)&&(x<3.210))
	{
		c0 =   342.7;
		c1 =    18.7;
		c2 =     0.0;

	}else if((x>=3.210)&&(x<4.038))
	{
		c0 =   352.2;
		c1 =    18.7;
		c2 =     0.0;

	}else if((x>=4.038)&&(x<7.111))
	{
		c0 =   433.9;
		c1 =    -2.4;
		c2 =     0.75;

	}else if((x>=7.111)&&(x<8.331))
	{
		c0 =   629.0;
		c1 =    30.9;
		c2 =     0.0;

	}else if((x>=8.331)&&(x<10.00))
	{
		c0 =   701.2;
		c1 =    25.2;
		c2 =     0.0;

	}else{	
		// extrapolate the > 10 keV results to higher frequencies
        c0 =   701.2;
        c1 =    25.2;
        c2 =     0.0;
    }
    // Use these coefficients to calculate the cross section per hydrogen atom
    return (1.0e-24)*(c0+c1*x+c2*x*x)/(x*x*x); //cm^2
}
