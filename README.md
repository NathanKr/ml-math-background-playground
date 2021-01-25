<h2>Motivation</h2>
python code to do basic math for machine learning

<table>
  <tr>
    <th>File</th>
    <th>Description</th>
  </tr>
  <tr>
    <td>gradient_descent_intercept.py</td>
    <td>
    <ul>
    <li>this is a code i did to follow https://www.youtube.com/watch?v=sDv4f4s2SB8 which fit a linear line based on 3 data points using gradient descent</li>
    <li>The data set is height vs weight shown on a plot. in machine learning this height is actually called y (here it is called observed_height) and weight is x</li>
    <li>the straight line equation is intercept + slope * weight and is called here predeicted_height , in machine learning this is called h</li>
    <li>The cost function here is ssr which sum the (observed[i] - predicted[i])^2 over all data points</li>
    <li>ssr is plot against the intercept for a given value of slope = 0.64 i.e intercept + 0.64 * weight</li>
    <li>the drivative of ssr with respect to intercept is computed and used by gradient descent to compute the value that minimize ssr. The observerd_height (h) is shown per iteration including ssr , step which decrease in every iteration</li>
    <li>it is shown very nicely that when gradient descent is far from the minimal cost the step is large ,but close to the minimal point the step is small</li>
    <li>the algorithm is stoped when step size is smaller that a common threshold of 0.001. in this case it took 8 iterations</li>
    </ul>
    </td>
  </tr>
  
</table>
