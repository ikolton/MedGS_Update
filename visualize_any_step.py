import streamlit as st
from PIL import Image
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from utils.general_utils import safe_state
from render_fast import render_set

# Streamlit UI configuration
st.set_page_config(
    page_title="Mesh Slicer Viewer",
    layout="wide",
)

st.title("Mesh Slicer Viewer")

st.markdown("""
    <style>
    .main { background-color: #fafafa; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    h1 { font-size: 2.5rem; }
    </style>
""", unsafe_allow_html=True)

parser = ArgumentParser(description="Testing script parameters")
model = ModelParams(parser, sentinel=True)
pipeline = PipelineParams(parser)

parser.add_argument("--iteration", default=-1, type=int)
parser.add_argument('--camera', type=str, default="mirror")
parser.add_argument("--distance", type=float, default=1.0)
parser.add_argument("--num_pts", type=int, default=100_000)
parser.add_argument("--skip_train", action="store_false")
parser.add_argument("--skip_test", action="store_true")
parser.add_argument("--quiet", action="store_true")
parser.add_argument("--poly_degree", type=int, default=1)
parser.add_argument("--interp", type=int, default=1)
parser.add_argument("--extension", type=str, default=".png")
parser.add_argument("--mask_path", type=str, default="-1")
parser.add_argument("--generate_points_path", type=str, default="-1")

args = get_combined_args(parser)

model.gs_type = "gs"
model.camera = args.camera
model.distance = args.distance
model.num_pts = args.num_pts
model.poly_degree = args.poly_degree

with st.spinner("Rendering..."):
    safe_state(args.quiet)
    rendered = render_sets(
        model.extract(args),
        args.iteration,
        pipeline.extract(args),
        args.skip_train,
        args.skip_test,
        args.interp,
        args.extension,
        generate_points_path=args.generate_points_path,
        mask_path=args.mask_path
    )

if isinstance(rendered, list) and len(rendered) >= 3:
    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(rendered[0], caption="Rendered Image 1", use_container_width=True)

    with col2:
        st.image(rendered[1], caption="Rendered Image 2", use_container_width=True)

    with col3:
        st.image(rendered[2], caption="Rendered Image 3", use_container_width=True)
else:
    st.warning("Expected at least 3 rendered images, but got something else.")

