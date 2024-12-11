#include <vector>
#include <cmath>
#include <stdexcept>

// Function to calculate Euclidean distance between two points.
double euclidean_distance(const std::vector<double>& point1, const std::vector<double>& point2) {
    if (point1.size() != point2.size()) {
        throw std::invalid_argument("Points must have the same dimensions.");
    }

    double distance = 0.0;
    for (size_t i = 0; i < point1.size(); ++i) {
        distance += std::pow(point1[i] - point2[i], 2);
    }
    return std::sqrt(distance);
}

// Function to compute distances from a target point to all points in a dataset.
std::vector<double> get_neighbor_distances(const std::vector<std::vector<double>>& data, const std::vector<double>& target_point) {
    std::vector<double> distances;
    for (const auto& point : data) {
        distances.push_back(euclidean_distance(point, target_point));
    }
    return distances;
}

// Function to calculate the volume of a hypersphere.
double calculate_hypersphere_volume(double radius, int dimensions) {
    if (dimensions < 1) {
        throw std::invalid_argument("Dimensions must be a positive integer.");
    }

    double pi = M_PI;
    double volume = std::pow(pi, dimensions / 2.0) * std::pow(radius, dimensions) / std::tgamma(dimensions / 2.0 + 1.0);
    return volume;
}

// Function to calculate density based on hypersphere approach.
std::vector<double> calculate_density(const std::vector<std::vector<double>>& data, double radius) {
    size_t n = data.size();
    if (n == 0) {
        throw std::invalid_argument("Data cannot be empty.");
    }

    int dimensions = data[0].size();
    double volume = calculate_hypersphere_volume(radius, dimensions);

    std::vector<double> densities;
    for (const auto& target_point : data) {
        int count = 0;
        for (const auto& point : data) {
            if (euclidean_distance(target_point, point) <= radius) {
                ++count;
            }
        }
        densities.push_back(count / volume);
    }

    return densities;
}
