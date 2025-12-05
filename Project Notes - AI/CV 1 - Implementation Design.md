
**ETH3D Multi-View Reconstruction – Implementation Overview**

This pipeline ingests the ETH3D `multi_view_training_dslr_undistorted` dataset and, per scene, builds a dense, occlusion-aware correspondence graph between views. A processing config initializes the `PipelineOrchestrator`, which uses `SceneManager` to discover scenes and load calibrated intrinsics, poses, points, and optional occlusion assets into a `SceneData` shell. Images are streamed through a GPU-accelerated preprocessor (color-space normalization, LAB/CLAHE, denoising, resizing), then `FeatureExtractor` computes SIFT features and per-image statistics that quantify coverage and texture quality. With camera poses available, the pipeline derives a view-overlap matrix and selects a compact set of high-value image pairs; for each pair, `FeatureMatcher` performs FLANN+ratio-test matching, robustifies with RANSAC geometry, and then prunes matches using 2D occlusion masks and optional 3D visibility checks against splats/meshes.

Surviving matches are fed into `CorrespondenceManager`, which merges pairwise correspondences into multi-view feature tracks, producing statistics over track length, visibility, and pose coverage. `VisualizationManager` generates targeted plots (feature density maps, overlap matrices, match quality panels, track histograms) when thresholds are violated or explicit debug flags are set. Finally, `ResultsManager` serializes all scene results—features, matches, tracks, scene metadata, and config snapshot—into HDF5/JSON plus plots under a timestamped results directory, giving a reproducible, analysis-ready artifact for PRCV experiments and downstream reconstruction or COLMAP integration.

```text
CONFIG → Orchestrator → SceneManager (ETH3D + occlusion)
      → GPU Preprocess → FeatureExtractor + ImageStats
      → ViewOverlap + PairSelection
      → Pose- & Occlusion-Aware Matching (FLANN + RANSAC)
      → CorrespondenceManager (multi-view tracks)
      → VisualizationManager (debug plots)
      → ResultsManager (HDF5 + JSON + figures)
```

## Implementation Design Flows – ETH3D Multi-View Correspondence Pipeline

The pipeline is orchestrated by `PipelineOrchestrator` and runs scene-wise over the ETH3D `multi_view_training_dslr_undistorted` dataset, with optional occlusion data. Here are the actual classes (`PipelineOrchestrator`, `SceneManager`, `GPUAcceleratedPreprocessor`, `FeatureExtractor`, `FeatureMatcher`, `CorrespondenceManager`, `ResultsManager`, etc.). The flow helps understand technical elements, inspections, research areas easily.

### 0. Entrypoint – Config → Orchestrator → Results

```text
CONFIG YAML / DICT
   └── ProcessingConfig.from_dict(...)
        ├── base_path        (ETH3D root)
        ├── save_dir         (results root, HDF5 + plots)
        ├── gpu_config       (use_gpu, device_id)
        ├── preprocessing    (batch_size, enhancements)
        ├── feature_detection (min_features, coverage thresholds)
        ├── matching_config  (view overlap, RANSAC, ratio test)
        └── visualization_config (force_all_plots, thresholds)
              ↓
PipelineOrchestrator(config)
   └── process_dataset()
        ├── SceneManager.discover_scenes()  → scene_list
        ├── select scenes (selected_scenes / num_scenes / scene_complexity)
        ├── per-scene: process_scene_wvisz(scene_name)
        └── aggregate into ReconstructionResults
                ↓
ResultsManager.save_computed_results(scene_data_by_scene, config)
   └── matches.h5 / features.h5 / meta.json / plots under save_dir/<timestamp>/
   └── ResultsRegistry.store(ReconstructionResults(...))
```

---

### 1. Scene + Metadata Init Module (ETH3D + Occlusion)

Goal: resolve ETH3D folder structure into a fully populated `SceneData` shell with intrinsics, poses, points, and occlusion.

```text
ETH3D BASE PATH (ProcessingConfig.base_path)
   └── SceneManager(base_path)
        ├── _discover_scenes()
        │     mv_undistorted/multi_view_training_dslr_undistorted/<scene>/
        └── _load_scene_metadata()
               ↓
for scene_name in selected_scenes:
    SCENE PATHS
      images_dir      ← mv_undistorted/.../<scene>/dslr_images_undistorted/
      calib_dir       ← mv_undistorted/.../<scene>/dslr_calibration_undistorted/
      occlusion_dir   ← mv_occlusion/.../<scene>/
          ↓
    CALIBRATION + STRUCTURE
      _load_camera_intrinsics(calib_dir)  → camera_intrinsics[id]: CameraIntrinsics
      _load_camera_poses(calib_dir)      → camera_poses[image_id] (quaternion + position)
      _load_points3d(calib_dir)          → points3D[id]: Point3D
      _load_points2D(calib_dir)          → points2D[image_id]: [Point2D]
      _load_image_camera_mapping(...)    → img_name → camera_id
    
    OCCLUSION DATA
      _load_occlusion_masks(scene_name)  → occlusion_masks[image_id]: OcclusionData
      _load_scene_occlusion(scene_name)  → SceneOcclusion(splats_path, mesh_path)

    META PACKING
      _get_scene_info(scene_name) → scene_meta = {
          "image_paths": [Path(JPG)...],
          "camera_intrinsics": {id → fx,fy,cx,cy},
          "camera_poses": {image_id → pose},
          "points3d": {id → XYZ,color,track_len},
          "points2d": {image_id → [(x,y,point3d_id), ...]},
          "occlusion_masks": {image_id → mask},
          "scene_occlusion": SceneOcclusion | None,
          "scene_complexity": estimated label,
          "has_poses": bool,
          "has_occlusion": bool,
      }

    VALIDATION + TYPE
      ValidationHelper.validate_scene_data(scene_meta)  → structural checks
      PipelineOrchestrator._validate_scene_meta(...)    → early exit on failure
      scene_type = _determine_scene_type(scene_meta)
          ↓
    INITIALIZE SCENEDATA
      SceneData.create_empty(scene_name, scene_meta, scene_type)
          ├── metadata     ← scene_meta
          ├── features     = {}
          ├── matches      = {}
          ├── correspondences = {}
          ├── matching_stats  = {}
          ├── occlusion_data  = occlusion_masks
          ├── scene_occlusion = SceneOcclusion
          └── stats placeholders (num_cameras, num_points, etc.)
```

Key switches:

* `ProcessingConfig.scene_complexity` + `MatchingConfig.distance_thresholds / angle_thresholds`
  dynamically change matching window and geometry thresholds per scene type.

---

### 2. GPU Preprocessing + Image Stats (Batch Path)

Goal: normalize huge DSLR images, apply enhancement on GPU, and compute per-image statistics.

```text
INPUT (per scene):
    image_paths   = scene_data.metadata["image_paths"]
    camera_poses  = scene_data.metadata["camera_poses"]

IMAGE DATASET
    dataset = ImageDataset(image_paths=image_paths,
                           camera_poses=camera_poses)
    loader  = torch.utils.data.DataLoader(dataset,
                 batch_size=preprocessing.batch_size,
                 num_workers=0, pin_memory=False, shuffle=False)

GPU PREPROCESSING (per batch)
    for batch in loader:
        batch_tensor = batch.to(device)   # 'cuda' if GPUConfig.use_gpu
            ↓
        GPUAcceleratedPreprocessor.preprocess_batch(batch_tensor)
            ├── BGR uint8 → float32 tensor [0,1]
            ├── RGB → XYZ → LAB (pre-computed matrices on GPU)
            ├── L-channel CLAHE (clipLimit=2.0, tile=8×8) on CPU
            ├── bilateral filter (d=5, σ_color=75, σ_space=75)
            ├── resize / reference size normalization (INTER_AREA)
            └── clamp + return (B,C,H,W) tensor on device

SINGLE IMAGE PROCESSING
    for img in processed_batch:
        img_path = dataset.image_paths[idx]
        img_pose = dataset.camera_poses.get(img_path.name)

        _process_single_image_wvisz(img, img_path, pose=img_pose)
            ├── tensor → numpy (H,W,3) for OpenCV
            ├── FeatureExtractor.extract_features(img_np)
            │       - SIFT_create(nfeatures, contrastThreshold, edgeThreshold, sigma)
            │       - returns { keypoints: List[cv2.KeyPoint], descriptors: np.ndarray }
            ├── FeatureExtractor.analyze_feature_quality(keypoints)
            │       - keypoint distribution grid
            │       - coverage %, entropy, blur cues
            ├── ImageStatsManager.compute_image_stats(img_np, keypoints, img_name)
            │       - feature count, coverage heatmap, histograms
            └── VisualizationManager.visualize_features(...) [gated]
                    - triggered if:
                        visualization_counter % interval == 0
                        OR feature_count below visualization_config.feature_threshold

SCENEDATA FILL
    scene_data.features[img_name] = ImageFeatures(
        keypoints   = keypoints,
        descriptors = descriptors,
        image_stats = image_stats
    )
    Scene-level aggregations:
        - total_features, avg_features_per_image
        - mean coverage, feature_density histograms
        - ImageStatsManager.plot_feature_density(...) (if show_feature_density)

Result: `SceneData.features` fully populated + `SceneStats` updated.
```

Important controls:

* `FeatureDetectionSettings.min_features`, `feature_density_threshold`
  decide whether an image is considered usable.
* `VisualizationConfig.feature_threshold`, `force_all_plots`, `show_feature_density`
  regulate heavy plotting.

---

### 3. Pose-Guided Pair Selection + View Overlap

Goal: choose a small, high-value set of image pairs for matching using camera poses and scene geometry.

```text
INPUT:
    scene_data.features        (per-image SIFT features)
    scene_data.metadata
        ├── camera_poses      (image_id → pose)
        └── camera_intrinsics

STEP 1 – PREPARE IMAGE DATA
    _prepare_image_data(scene_data.features, images, scene_data)  → image_data[]
        each entry:
           {
             "image_id": img_id,
             "features": ImageFeatures,
             "pose": camera_poses[img_id] | None,
             "intrinsics": camera_intrinsics[camera_id] | None,
             "occlusion_mask": occlusion_data[img_id] | None,
           }

STEP 2 – VIEW OVERLAP MATRIX
    if metadata["has_poses"]:
        overlap_matrix = FeatureMatcher.compute_view_overlap_matrix(image_data)
            ├── uses quaternions + positions
            ├── estimates frustum intersection + baseline ratios
            └── produces symmetric [N×N] overlap_score matrix

        VisualizationManager.visualize_view_overlap(camera_poses, overlap_matrix)
            - color-coded overlaps, sequential vs non-sequential pairs

STEP 3 – SELECT MATCHING PAIRS
    selected_pairs = _get_matching_pairs(image_data, matching_stats)
        ├── apply MatchingConfig:
        │     - max_distance_threshold
        │     - min_overlap_threshold
        │     - max_neighbors per image
        ├── prefer sequential neighbors (i, i+1, i+2)
        └── record candidate pairs into matching_stats["total_pairs"]

Result: compact list of image index pairs `(idx1, idx2)` to feed into matching.
```

---

### 4. Feature Matching + Occlusion-Aware Filtering

Goal: run FLANN+RANSAC matching, then prune using 2D masks and 3D occlusion checks.

```text
for (idx1, idx2) in selected_pairs:

    img1_id, img2_id = images[idx1], images[idx2]
    features1, features2 = scene_data.features[img1_id], scene_data.features[img2_id]
    pose1, pose2 = metadata["camera_poses"].get(img1_id), metadata["camera_poses"].get(img2_id)

    # 4.1 – Core Matching
    match_results = FeatureMatcher.match_features(
                        features1.descriptors, features2.descriptors,
                        pose1=pose1, pose2=pose2, scene_type=scene_data.scene_type)
        ├── FLANN (KD-Tree) knnMatch(k=2)
        ├── Lowe ratio test (ratio_test_threshold)
        ├── optional symmetry checks / descriptor similarity filters
        ├── RANSAC fundamental matrix:
        │       F, inlier_mask = cv2.findFundamentalMat(
        │            pts1, pts2, method=cv2.FM_RANSAC,
        │            ransacReprojThreshold=ransac_threshold,
        │            confidence=confidence)
        └── outputs:
                {
                  "verified_matches": inlier_matches,
                  "fundamental_matrix": F,
                  "inlier_ratio": (#inliers / #matches),
                  "pose_guided": bool,
                  "visibility_stats": {...}  (filled after occlusion),
                }

    # 4.2 – Occlusion-Aware Filtering
    filtered_matches = _filter_matches_by_occlusion(
                           scene_data, img1_id, img2_id,
                           match_results["verified_matches"],
                           features1, features2)
        ├── _check_2d_mask_visibility(...)
        │     - project keypoints into occlusion_mask[image_id]
        │     - require visibility > visibility_threshold_2d
        ├── _check_3d_visibility(...)
        │     - use points3D + SceneOcclusion (splat/mesh)
        │     - check ray intersections, baseline angles
        └── update visibility_stats: {total, passed_2d, passed_3d, passed_both}

    match_results["verified_matches"] = filtered_matches

    # 4.3 – Pose Metrics + Logging
    if metadata["has_poses"]:
        pose_metrics = SceneManager.compute_pose_metrics(
                           scene_name, img1_id, img2_id,
                           matches=filtered_matches,
                           kp1=features1.keypoints, kp2=features2.keypoints)
        match_results["pose_metrics"] = pose_metrics

    scene_data.matches[(img1_id, img2_id)] = match_results

    # 4.4 – Statistics + Visualization Triggers
    _update_match_statistics_new(match_results, matching_stats, idx1, idx2)
        ├── total_matches, sequential vs non-sequential counts
        ├── average_matches_per_pair
        ├── descriptor_metrics.average_match_distance
        ├── descriptor_metrics.inlier_ratios[]
        └── pose_guided_matches, pose_rejection_rate

    _check_visualization_triggers(...)
        └── VisualizationManager.visualize_matches(
                img1, img2, features1.keypoints, features2.keypoints,
                filtered_matches, title="Match Quality Plots ...")
           - forced if inlier_ratio < match_threshold
           - or config.force_all_plots == True
```

---

### 5. Multi-View Track Building + Final Outputs

Goal: convert pairwise matches into multi-view feature tracks and write everything to disk in HDF5 + JSON.

```text
INITIALIZE TRACKING
    _initialize_correspondence_tracking(scene_data)
        └── for each img_id in scene_data.features:
                CorrespondenceManager.initialize_image(img_id, ImageFeatures)
                # sets up track_mappings per image

DURING MATCH PROCESSING
    CorrespondenceManager.add_matches(img1_id, img2_id, filtered_matches)
        ├── map (queryIdx, trainIdx) → shared track_id
        ├── merge tracks if same 3D point appears across pairs
        └── maintain:
                tracks[track_id] = { image_id → feature_index }

FINALIZE TRACKS
    _finalize_and_visualize_tracks(scene_data)
        ├── min_len = CorrespondenceManager.get_track_length_threshold()
        ├── tracks = CorrespondenceManager.get_feature_tracks(min_length=min_len)
        ├── stats  = CorrespondenceManager.get_track_length_statistics()
        ├── scene_data.correspondences = {
                "tracks": tracks,
                "track_statistics": stats,
            }
        └── VisualizationManager.visualize_tracks(...)
                - track length histograms
                - overlayed 2D/3D tracks, camera graph

MATCHING SUMMARY
    _finalize_matching_stats(matching_stats)
        ├── normalize averages
        ├── compute visibility & pose coverage aggregates
        └── store into scene_data.matching_stats

SCENE-LEVEL PACKING
    ReconstructionResults.scene_data[scene_name] = scene_data
        ├── metadata (intrinsics, poses, points3D, points2D, complexity)
        ├── features   (ImageFeatures per image)
        ├── matches    (pair → match_results)
        ├── correspondences (tracks + stats)
        ├── matching_stats
        └── scene_stats (feature coverage, view overlap, pose metrics)

PERSISTENCE (ResultsManager)
    save_computed_results(scene_data_by_scene, config)
        ├── _convert_results_for_saving(...)  → storage_format (pure dict)
        ├── write features.h5 (per-image keypoints + descriptors)
        ├── write matches.h5  (per-pair matches + RANSAC outcomes)
        ├── write correspondences.h5 (tracks + per-track stats)
        ├── write meta.json (config snapshot, scene summaries)
        └── store plots under <save_dir>/<timestamp>/
               - feature density, overlap matrices, match quality, tracks
```

---

### 6. Unfinished: COLMAP Integration Path (PyColmapProcessor)

(Currently isolated module; can be wired after correspondences.)

```text
PyColmapConfig        → tuning mapper / BA options
   ↓
PyColmapProcessor(config)
   ├── run_incremental_mapper(image_dir, database_path, output_path)
   ├── bundle_adjustment(...)
   └── export_reconstruction(...)
         ↓
COLMAP sparse model (cameras.bin, images.bin, points3D.bin)
   ↔  SceneData.metadata.{camera_intrinsics, camera_poses, points3D}
      (bridge layer can translate between PyColmap and SceneData)
```
