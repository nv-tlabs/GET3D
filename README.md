## GET3D: A Generative Model of High Quality 3D Textured Shapes Learned from Images (NeurIPS 2022)<br><sub>Official PyTorch implementation </sub>

![Teaser image](./docs/assets/get3d_model.png)

**GET3D: A Generative Model of High Quality 3D Textured Shapes Learned from Images**<br>
[Jun Gao](http://www.cs.toronto.edu/~jungao/), [Tianchang Shen](http://www.cs.toronto.edu/~shenti11/), [Zian Wang](http://www.cs.toronto.edu/~zianwang/), 
[Wenzheng Chen](http://www.cs.toronto.edu/~wenzheng/), [Kangxue Yin](https://kangxue.org/), [Daiqing Li](https://scholar.google.ca/citations?user=8q2ISMIAAAAJ&hl=en), 
[Or Litany](https://orlitany.github.io/), [Zan Gojcic](https://zgojcic.github.io/), 
[Sanja Fidler](https://www.cs.toronto.edu/~fidler/) <br>
**[Paper](https://nv-tlabs.github.io/GET3D/assets/paper.pdf), [Project Page](https://nv-tlabs.github.io/GET3D/)**

Abstract: *As several industries are moving towards modeling massive 3D virtual worlds, the need for content creation
tools that can scale in terms of the quantity, quality, and diversity of 3D content is becoming evident. In our work, we
aim to train performant 3D generative models that synthesize textured meshes which can be directly consumed by 3D
rendering engines, thus immediately usable in downstream applications. Prior works on 3D generative modeling either lack
geometric details, are limited in the mesh topology they can produce, typically do not support textures, or utilize
neural renderers in the synthesis process, which makes their use in common 3D software non-trivial. In this work, we
introduce GET3D, a Generative model that directly generates Explicit Textured 3D meshes with complex topology, rich
geometric details, and high fidelity textures. We bridge recent success in the differentiable surface modeling,
differentiable rendering as well as 2D Generative Adversarial Networks to train our model from 2D image collections.
GET3D is able to generate high-quality 3D textured meshes, ranging from cars, chairs, animals, motorbikes and human
characters to buildings, achieving significant improvements over previous methods.*


![Teaser Results](./docs/assets/teaser_result.jpg)

## News

- 2022-09-22: Code will be uploaded next week!

## License

Copyright &copy; 2022, NVIDIA Corporation & affiliates. All rights reserved.

This work is made available under
the [Nvidia Source Code License](https://github.com/nv-tlabs/GET3D/blob/master/LICENSE.txt).

## Citation

```latex
@inproceedings{gao2022get3d,
    title={GET3D: A Generative Model of High Quality 3D Textured Shapes Learned from Images},
    author={Jun Gao and Tianchang Shen and Zian Wang and Wenzheng Chen and Kangxue Yin 
        and Daiqing Li and Or Litany and Zan Gojcic and Sanja Fidler},
    booktitle={Advances In Neural Information Processing Systems},
    year={2022}
}
```