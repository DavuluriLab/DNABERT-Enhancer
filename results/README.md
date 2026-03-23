## Model performance
DNABERT-Enhancer includes two models namely, DNABERT-Enhancer-201 and DNABERT-Enhancer-350, trained on enhancer data of 201 bp and 350 bp respectively. The models were compared to the Base-line methods, Recent enhancer prediction methods and the Nucleotide traansformer.

### Comparison to Base-line methods

<table align="center">
  <thead>
    <tr>
      <th rowspan="2">Models</th>
      <th colspan="5">Eds-201-Test</th>
      <th colspan="5">Eds-350-Test</th>
    </tr>
    <tr>
      <th>Accuracy (%)</th>
      <th>Precision (%)</th>
      <th>Recall (%)</th>
      <th>F1 score (%)</th>
      <th>MCC (%)</th>
      <th>Accuracy (%)</th>
      <th>Precision (%)</th>
      <th>Recall (%)</th>
      <th>F1 score (%)</th>
      <th>MCC (%)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Random Forest</td>
      <td>74.74</td>
      <td>75.36</td>
      <td>73.51</td>
      <td>74.42</td>
      <td>49.49</td>
      <td>78.12</td>
      <td>79.49</td>
      <td>75.81</td>
      <td>77.61</td>
      <td>56.31</td>
    </tr>
    <tr>
      <td>KNN</td>
      <td>57.1</td>
      <td>82.01</td>
      <td>18.19</td>
      <td>29.78</td>
      <td>22.62</td>
      <td>53.63</td>
      <td>95.4</td>
      <td>7.64</td>
      <td>14.14</td>
      <td>18.54</td>
    </tr>
    <tr>
      <td>SVC</td>
      <td>73.47</td>
      <td>73.56</td>
      <td>73.28</td>
      <td>73.42</td>
      <td>46.95</td>
      <td>78.07</td>
      <td>77.47</td>
      <td>79.15</td>
      <td>78.3</td>
      <td>56.14</td>
    </tr>
    <tr>
      <td>Gaussian Naive Bayes</td>
      <td>69.56</td>
      <td>71.2</td>
      <td>65.71</td>
      <td>68.34</td>
      <td>39.24</td>
      <td>72.58</td>
      <td>74.42</td>
      <td>68.83</td>
      <td>71.51</td>
      <td>45.3</td>
    </tr>
    <tr>
      <td>AdaBoost Classifier</td>
      <td>75.73</td>
      <td>75.4</td>
      <td>76.38</td>
      <td>75.89</td>
      <td>51.46</td>
      <td>77.9</td>
      <td>77.8</td>
      <td>78.07</td>
      <td>77.93</td>
      <td>55.8</td>
    </tr>
    <tr>
      <td>MLP</td>
      <td>69.24</td>
      <td>68.99</td>
      <td>69.9</td>
      <td>69.45</td>
      <td>38.49</td>
      <td>77.2</td>
      <td>76.84</td>
      <td>77.88</td>
      <td>77.36</td>
      <td>54.41</td>
    </tr>
    <tr>
      <td><b>DNABERT-Enhancer</b></td>
      <td><b>82.04</b></td>
      <td><b>84.64</b></td>
      <td><b>78.29</b></td>
      <td><b>81.3</b></td>
      <td><b>64.27</b></td>
      <td><b>88.05</b></td>
      <td><b>90.27</b></td>
      <td><b>85.29</b></td>
      <td><b>87.71</b></td>
      <td><b>76.22</b></td>
    </tr>
  </tbody>
</table>
