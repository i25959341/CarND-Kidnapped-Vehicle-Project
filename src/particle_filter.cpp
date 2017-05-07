/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <unordered_map>

#include "particle_filter.h"

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	num_particles=1000;

	double std_x, std_y, std_theta;

	std_x = std[0];
	std_y =std[1];
	std_theta = std[2];

	std::normal_distribution<double> dist_x(x, std_x);
	std::normal_distribution<double> dist_y(y, std_y);
	std::normal_distribution<double> dist_psi(theta, std_theta);
	weights.resize(num_particles);

	for (int i = 0; i < num_particles; ++i) {
		double sample_x, sample_y, sample_theta;

		sample_x = dist_x(gen);
		sample_y = dist_y(gen);
		sample_theta = dist_psi(gen);

		Particle particle;
		particle.id=i;
		particle.x=sample_x;
		particle.y=sample_y;
		particle.theta=sample_theta;
		particle.weight=1.0;
        weights.push_back(particle.weight);

		particles.push_back(particle);
	}
	is_initialized=true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

    double std_x, std_y, std_theta;
    std_x = std_pos[0];
    std_y =std_pos[1];
    std_theta = std_pos[2];

    for (int i = 0; i < num_particles; ++i) {
        double x, y, theta;

        x = particles.at(i).x;
        y = particles.at(i).y;
        theta = particles.at(i).theta;

        if(yaw_rate==0.0){
            x = x + velocity*cos(theta)*delta_t;
            y = y + velocity*sin(theta)*delta_t;
        }else{
            x = x + velocity/yaw_rate*(sin(theta+yaw_rate*delta_t)-sin(theta));
            y = y + velocity/yaw_rate*(cos(theta)-cos(theta+yaw_rate*delta_t));
            theta = theta+yaw_rate*delta_t;
        }

        std::normal_distribution<double> dist_x(x, std_x);
        std::normal_distribution<double> dist_y(y, std_y);
        std::normal_distribution<double> dist_psi(theta, std_theta);

        x = x + dist_x(gen)*delta_t;
        y = y + dist_y(gen)*delta_t;
        theta = theta + dist_psi(gen)*delta_t;

        particles.at(i).x = x;
        particles.at(i).y = y;
        particles.at(i).theta = theta;
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
    for(LandmarkObs& obs:observations){
        double min_distance = std::numeric_limits<double>::max();
        for (LandmarkObs& predict:predicted){
            double distance = dist(obs.x,obs.y,predict.x,predict.y);
            if(distance<min_distance){
                min_distance=distance;
                obs.id=predict.id;
            }
        }
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account 
	//   for the fact that the map's y-axis actually points downwards.)
	//   http://planning.cs.uiuc.edu/node99.html

    double sigma_x, sigma_y, sigma_x_2, sigma_y_2, sigma_xy;
    sigma_x = std_landmark[0];
    sigma_y = std_landmark[1];
    sigma_x_2 = pow(sigma_x, 2);
    sigma_y_2 = pow(sigma_y, 2);
    sigma_xy = 2*M_PI*sigma_x*sigma_y;

    // For each particles

    for (int i =0; i<particles.size();i++){
        Particle particle = particles[i];
        double x = particle.x;
        double y = particle.y;
        double theta = particle.theta;

        std::vector<LandmarkObs> predicted;
        std::unordered_map<int, LandmarkObs> pred_map;
        for(Map::single_landmark_s landmark : map_landmarks.landmark_list){
            if(dist(landmark.x_f,landmark.y_f,x,y)<sensor_range){
                LandmarkObs pred;
                pred.x =landmark.x_f;
                pred.y=landmark.y_f;
                pred.id = landmark.id_i;
                predicted.push_back(pred);
                pred_map[pred.id] = pred;
            }
        }

        // Transfrom observation into Map coordinates
        std::vector<LandmarkObs> trasformedObs;
        for (LandmarkObs obs : observations){
            double x_new;
            double y_new;

            x_new = x*cos(theta) - y*sin(theta) + x;
            y_new = x*sin(theta) + y*cos(theta) + y;
            LandmarkObs trasformedObservation;
            trasformedObservation.x=x_new;
            trasformedObservation.y=y_new;
            trasformedObs.push_back(trasformedObservation);
        }

        // Do data association for this particular particle
        dataAssociation(predicted, trasformedObs);

        // Calculate the gaussian probablity
        std::vector<double> gaussians;

        double total_weights=1

        for(LandmarkObs obs:trasformedObs){
            LandmarkObs pred =pred_map[obs.id];

            double dx = obs.x-pred.x;
            double dy = obs.y-pred.y;

            double diff_x_2 = dx*dx;
            double diff_y_2 = dy*dy;

            double weight = exp(-(diff_x_2/(2*sigma_x_2) + diff_y_2/(2*sigma_y_2)))/(sigma_xy);

            total_weights=total_weights*weight;

        }
        // Update the weight
        particle.weight = total_weights;
        weights[i] = total_weights;
    }

}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

}

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}
