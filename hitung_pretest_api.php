<?php
if ($_SERVER['REQUEST_METHOD'] == 'GET' && realpath(__FILE__) == realpath($_SERVER['SCRIPT_FILENAME'])) {
    header('HTTP/1.0 403 Forbidden', TRUE, 403);
    die(header('location: ../index.php'));
}

include('../config/db.php');
session_start();

// Menghitung modus (sama seperti file asli)
function mode($armodul)
{
    $v = array_count_values($armodul);
    $total = 0;
    arsort($v);
    foreach ($v as $k => $v) {
        $total = $k;
        break;
    }
    return $total;
}

// Fungsi untuk memanggil GAT API batch
function callGATBatchAPI($students_data, $api_url = 'https://your-vercel-app.vercel.app/predict-batch') {
    $curl = curl_init();
    
    $post_data = json_encode([
        'students' => $students_data
    ]);
    
    curl_setopt_array($curl, [
        CURLOPT_URL => $api_url,
        CURLOPT_RETURNTRANSFER => true,
        CURLOPT_ENCODING => '',
        CURLOPT_MAXREDIRS => 10,
        CURLOPT_TIMEOUT => 30,
        CURLOPT_FOLLOWLOCATION => true,
        CURLOPT_HTTP_VERSION => CURL_HTTP_VERSION_1_1,
        CURLOPT_CUSTOMREQUEST => 'POST',
        CURLOPT_POSTFIELDS => $post_data,
        CURLOPT_HTTPHEADER => [
            'Content-Type: application/json',
            'Content-Length: ' . strlen($post_data)
        ],
    ]);
    
    $response = curl_exec($curl);
    $http_code = curl_getinfo($curl, CURLINFO_HTTP_CODE);
    $error = curl_error($curl);
    
    curl_close($curl);
    
    if ($error) {
        return [
            'success' => false,
            'error' => 'cURL Error: ' . $error
        ];
    }
    
    if ($http_code !== 200) {
        return [
            'success' => false,
            'error' => 'HTTP Error: ' . $http_code,
            'response' => $response
        ];
    }
    
    $decoded = json_decode($response, true);
    if (json_last_error() !== JSON_ERROR_NONE) {
        return [
            'success' => false,
            'error' => 'JSON Decode Error: ' . json_last_error_msg(),
            'response' => $response
        ];
    }
    
    return [
        'success' => true,
        'data' => $decoded
    ];
}

// Submit data pre-test dengan GAT API
if ($_SERVER["REQUEST_METHOD"] == "POST") {

    // ========== BAGIAN 1: AMBIL DATA SISWA DAN JAWABAN (SAMA SEPERTI ASLI) ==========
    
    // Mengambil data murid sesuai kelas
    $sql = "SELECT id FROM student where class_id = '{$_POST['id']}'  order by id ASC";
    $query = mysqli_query($conn, $sql);
    $students = mysqli_fetch_all($query, MYSQLI_ASSOC);
    $murid = [];
    $i = 0;

    // perulangan murid untuk mengambil data jawaban pre_test
    foreach ($students as $key => $s) {
        $i++;
        $sql = "SELECT * FROM pre_test_answer WHERE student_id = '{$s['id']}'";
        $query = mysqli_query($conn, $sql);
        $answer = mysqli_fetch_array($query, MYSQLI_ASSOC);
        
        if ($answer) {
            $murid[$s['id']] = array(
                'modul_1' => $answer['modul_1'],
                'modul_2' => $answer['modul_2'],
                'modul_3' => $answer['modul_3'],
                'modul_4' => $answer['modul_4'],
                'modul_5' => $answer['modul_5'],
                'modul_6' => $answer['modul_6'],
                'modul_7' => $answer['modul_7'],
            );
        }
    }

    if (empty($murid)) {
        echo "<p style='color: red;'>❌ Tidak ada data jawaban pre-test untuk kelas ini.</p>";
        exit;
    }

    echo "<h3>📊 Memproses Pre-test dengan IRT + GAT API...</h3>";
    echo "<p><strong>Kelas ID:</strong> {$_POST['id']}</p>";
    echo "<p><strong>Jumlah siswa:</strong> " . count($murid) . "</p>";
    echo "<hr>";

    // ========== BAGIAN 2: IRT PROCESSING (COPY EXACT DARI FILE ASLI) ==========
    
    $sum_modul_1 = array();
    $sum_modul_2 = array();
    $sum_modul_3 = array();
    $sum_modul_4 = array();
    $sum_modul_5 = array();
    $sum_modul_6 = array();
    $sum_modul_7 = array();

    // Menghitung P dan ability (EXACT DARI FILE ASLI)
    foreach ($murid as $key => $m) {
        ${'p_user_' . $key} = array_sum($m) / count($m);
        $murid[$key]['p_user'] = ${'p_user_' . $key};
        if ($murid[$key]['p_user'] == 0) {
            $murid[$key]['p_user'] = 0.1;
        } else if ($murid[$key]['p_user'] == 1) {
            $murid[$key]['p_user'] = 0.9;
        }
        $ability = log($murid[$key]['p_user'] / (1 - $murid[$key]['p_user']));
        $murid[$key]['ability'] = $ability;
        
        $sum_modul_1[] = $murid[$key]['modul_1'];
        $sum_modul_2[] = $murid[$key]['modul_2'];
        $sum_modul_3[] = $murid[$key]['modul_3'];
        $sum_modul_4[] = $murid[$key]['modul_4'];
        $sum_modul_5[] = $murid[$key]['modul_5'];
        $sum_modul_6[] = $murid[$key]['modul_6'];
        $sum_modul_7[] = $murid[$key]['modul_7'];
    }

    // HITUNG P Modul (EXACT DARI FILE ASLI)
    $p_modul_1 = array_sum($sum_modul_1) / count($sum_modul_1);
    $p_modul_2 = array_sum($sum_modul_2) / count($sum_modul_2);
    $p_modul_3 = array_sum($sum_modul_3) / count($sum_modul_3);
    $p_modul_4 = array_sum($sum_modul_4) / count($sum_modul_4);
    $p_modul_5 = array_sum($sum_modul_5) / count($sum_modul_5);
    $p_modul_6 = array_sum($sum_modul_6) / count($sum_modul_6);
    $p_modul_7 = array_sum($sum_modul_7) / count($sum_modul_7);

    // Hitung difficulty Modul (EXACT DARI FILE ASLI)
    if ($p_modul_1 == 0) { $p_modul_1 = 1; }
    if ($p_modul_2 == 0) { $p_modul_2 = 1; }
    if ($p_modul_3 == 0) { $p_modul_3 = 1; }
    if ($p_modul_4 == 0) { $p_modul_4 = 1; }
    if ($p_modul_5 == 0) { $p_modul_5 = 1; }
    if ($p_modul_6 == 0) { $p_modul_6 = 1; }
    if ($p_modul_7 == 0) { $p_modul_7 = 1; }
    
    $difficulty_modul_1 = log((1 - $p_modul_1) / $p_modul_1);
    $difficulty_modul_2 = log((1 - $p_modul_2) / $p_modul_2);
    $difficulty_modul_3 = log((1 - $p_modul_3) / $p_modul_3);
    $difficulty_modul_4 = log((1 - $p_modul_4) / $p_modul_4);
    $difficulty_modul_5 = log((1 - $p_modul_5) / $p_modul_5);
    $difficulty_modul_6 = log((1 - $p_modul_6) / $p_modul_6);
    $difficulty_modul_7 = log((1 - $p_modul_7) / $p_modul_7);

    $difficultys = [
        $difficulty_modul_1, $difficulty_modul_2, $difficulty_modul_3, $difficulty_modul_4, $difficulty_modul_5, $difficulty_modul_6, $difficulty_modul_7
    ];

    //hitung average difficulty
    $difficulty_average = array_sum($difficultys) / count($difficultys);

    //hitung adjust difficulty
    $adj_difficulty_modul_1 = $difficulty_modul_1 - $difficulty_average;
    $adj_difficulty_modul_2 = $difficulty_modul_2 - $difficulty_average;
    $adj_difficulty_modul_3 = $difficulty_modul_3 - $difficulty_average;
    $adj_difficulty_modul_4 = $difficulty_modul_4 - $difficulty_average;
    $adj_difficulty_modul_5 = $difficulty_modul_5 - $difficulty_average;
    $adj_difficulty_modul_6 = $difficulty_modul_6 - $difficulty_average;
    $adj_difficulty_modul_7 = $difficulty_modul_7 - $difficulty_average;

    $adj_difficultys = [
        $adj_difficulty_modul_1, $adj_difficulty_modul_2, $adj_difficulty_modul_3, $adj_difficulty_modul_4, $adj_difficulty_modul_5, $adj_difficulty_modul_6, $adj_difficulty_modul_7
    ];

    // ========== BAGIAN 3: ITERASI IRT (SIMPLIFIED - CUKUP SAMPAI ITERASI BEBERAPA KALI) ==========
    
    echo "<p>🔄 Menjalankan IRT iterasi...</p>";
    
    // Hitung iterasi pertama (EXACT DARI FILE ASLI)
    $iterasi1_harap = array();
    foreach ($murid as $key => $m) {
        $iterasi1_harap[$key] = array(
            'modul_1' => exp($murid[$key]['ability'] - $adj_difficulty_modul_1) / (1 + exp($murid[$key]['ability'] - $adj_difficulty_modul_1)),
            'modul_2' => exp($murid[$key]['ability'] - $adj_difficulty_modul_2) / (1 + exp($murid[$key]['ability'] - $adj_difficulty_modul_2)),
            'modul_3' => exp($murid[$key]['ability'] - $adj_difficulty_modul_3) / (1 + exp($murid[$key]['ability'] - $adj_difficulty_modul_3)),
            'modul_4' => exp($murid[$key]['ability'] - $adj_difficulty_modul_4) / (1 + exp($murid[$key]['ability'] - $adj_difficulty_modul_4)),
            'modul_5' => exp($murid[$key]['ability'] - $adj_difficulty_modul_5) / (1 + exp($murid[$key]['ability'] - $adj_difficulty_modul_5)),
            'modul_6' => exp($murid[$key]['ability'] - $adj_difficulty_modul_6) / (1 + exp($murid[$key]['ability'] - $adj_difficulty_modul_6)),
            'modul_7' => exp($murid[$key]['ability'] - $adj_difficulty_modul_7) / (1 + exp($murid[$key]['ability'] - $adj_difficulty_modul_7)),
        );
    }

    // ITERASI VARIAN (SIMPLIFIED VERSION)
    $iterasi1_varian = array();
    foreach ($murid as $key => $m) {
        $iterasi1_varian[$key]['modul_1'] = $iterasi1_harap[$key]['modul_1'] * (1 - $iterasi1_harap[$key]['modul_1']);
        $iterasi1_varian[$key]['modul_2'] = $iterasi1_harap[$key]['modul_2'] * (1 - $iterasi1_harap[$key]['modul_2']);
        $iterasi1_varian[$key]['modul_3'] = $iterasi1_harap[$key]['modul_3'] * (1 - $iterasi1_harap[$key]['modul_3']);
        $iterasi1_varian[$key]['modul_4'] = $iterasi1_harap[$key]['modul_4'] * (1 - $iterasi1_harap[$key]['modul_4']);
        $iterasi1_varian[$key]['modul_5'] = $iterasi1_harap[$key]['modul_5'] * (1 - $iterasi1_harap[$key]['modul_5']);
        $iterasi1_varian[$key]['modul_6'] = $iterasi1_harap[$key]['modul_6'] * (1 - $iterasi1_harap[$key]['modul_6']);
        $iterasi1_varian[$key]['modul_7'] = $iterasi1_harap[$key]['modul_7'] * (1 - $iterasi1_harap[$key]['modul_7']);
        $iterasi1_varian[$key]['sum'] = -1 * (array_sum($iterasi1_varian[$key]));
    }

    // ITERASI RESIDUAL (SIMPLIFIED VERSION)
    $iterasi1_residual = array();
    foreach ($murid as $key => $m) {
        $iterasi1_residual[$key]['modul_1'] = $murid[$key]['modul_1'] - $iterasi1_harap[$key]['modul_1'];
        $iterasi1_residual[$key]['modul_2'] = $murid[$key]['modul_2'] - $iterasi1_harap[$key]['modul_2'];
        $iterasi1_residual[$key]['modul_3'] = $murid[$key]['modul_3'] - $iterasi1_harap[$key]['modul_3'];
        $iterasi1_residual[$key]['modul_4'] = $murid[$key]['modul_4'] - $iterasi1_harap[$key]['modul_4'];
        $iterasi1_residual[$key]['modul_5'] = $murid[$key]['modul_5'] - $iterasi1_harap[$key]['modul_5'];
        $iterasi1_residual[$key]['modul_6'] = $murid[$key]['modul_6'] - $iterasi1_harap[$key]['modul_6'];
        $iterasi1_residual[$key]['modul_7'] = $murid[$key]['modul_7'] - $iterasi1_harap[$key]['modul_7'];
        $iterasi1_residual[$key]['sum_residual'] = array_sum($iterasi1_residual[$key]);
        $iterasi1_residual[$key]['new_ability'] = $murid[$key]['ability'] - $iterasi1_residual[$key]['sum_residual'] / $iterasi1_varian[$key]['sum'];
    }

    // Update ability dengan hasil iterasi pertama
    foreach ($murid as $key => $m) {
        $murid[$key]['final_ability'] = $iterasi1_residual[$key]['new_ability'];
    }

    echo "<p>✅ IRT iterasi selesai.</p>";

    // ========== BAGIAN 4: PANGGIL GAT API UNTUK BATCH PREDICTION ==========
    
    echo "<p>🤖 Menyiapkan data untuk GAT API...</p>";
    
    // Prepare data untuk GAT API
    $gat_students = [];
    foreach ($murid as $student_id => $student_data) {
        // Skip siswa dengan data kosong (EXACT DARI FILE ASLI)
        if ($student_data['modul_1'] == '' || $student_data['modul_2'] == '' || 
            $student_data['modul_3'] == '' || $student_data['modul_4'] == '' || 
            $student_data['modul_5'] == '' || $student_data['modul_6'] == '' || 
            $student_data['modul_7'] == '') {
            continue;
        }
        
        // Ambil survey confidence dari database (atau default 0.7)
        $sql = "SELECT level_result FROM survey_result WHERE student_id = '{$student_id}'";
        $query = mysqli_query($conn, $sql);
        $survey_result = mysqli_fetch_array($query, MYSQLI_ASSOC);
        
        // Convert survey level to confidence (1->0.9, 2->0.7, 3->0.5)
        $survey_confidence = 0.7; // default
        if ($survey_result && $survey_result['level_result']) {
            switch (intval($survey_result['level_result'])) {
                case 1: $survey_confidence = 0.9; break;
                case 2: $survey_confidence = 0.7; break;
                case 3: $survey_confidence = 0.5; break;
                default: $survey_confidence = 0.7; break;
            }
        }
        
        // Gunakan final ability dari IRT
        $final_ability = isset($student_data['final_ability']) ? $student_data['final_ability'] : $student_data['ability'];
        
        // Ensure ability is in valid range for GAT API (-5.0 to 5.0)
        $final_ability = max(-5.0, min(5.0, $final_ability));
        
        $gat_students[] = [
            'student_id' => strval($student_id),
            'irt_ability' => floatval($final_ability),
            'survey_confidence' => floatval($survey_confidence)
        ];
    }
    
    if (empty($gat_students)) {
        echo "<p style='color: red;'>❌ Tidak ada siswa dengan data lengkap untuk diproses GAT API.</p>";
        exit;
    }
    
    echo "<p>📤 Mengirim " . count($gat_students) . " siswa ke GAT API...</p>";
    
    // Panggil GAT API
    $api_result = callGATBatchAPI($gat_students);
    
    if (!$api_result['success']) {
        echo "<div style='color: red; border: 1px solid red; padding: 10px; margin: 10px 0;'>";
        echo "<h4>❌ Error calling GAT API:</h4>";
        echo "<p><strong>Error:</strong> " . $api_result['error'] . "</p>";
        if (isset($api_result['response'])) {
            echo "<p><strong>Response:</strong></p>";
            echo "<pre style='background: #f5f5f5; padding: 10px; overflow: auto;'>" . htmlspecialchars($api_result['response']) . "</pre>";
        }
        echo "</div>";
        exit;
    }
    
    $gat_response = $api_result['data'];
    
    if ($gat_response['status'] !== 'success') {
        echo "<p style='color: red;'>❌ GAT API returned error: " . $gat_response['message'] . "</p>";
        exit;
    }
    
    echo "<div style='color: green; border: 1px solid green; padding: 10px; margin: 10px 0;'>";
    echo "<h4>✅ GAT API Response Berhasil!</h4>";
    echo "<p>📈 <strong>Successful predictions:</strong> " . $gat_response['data']['successful_predictions'] . "</p>";
    echo "<p>❌ <strong>Failed predictions:</strong> " . $gat_response['data']['failed_predictions'] . "</p>";
    echo "</div>";

    // ========== BAGIAN 5: PROCESS RESULTS DAN SIMPAN KE DATABASE (EXACT SEPERTI FILE ASLI) ==========
    
    $processed_count = 0;
    $error_count = 0;
    
    echo "<h4>📋 Processing Results per Student:</h4>";
    
    foreach ($gat_response['data']['results'] as $student_result) {
        $student_id = $student_result['student_id'];
        $level_pre_test = $student_result['predicted_level'];
        $confidence = $student_result['confidence'];
        
        echo "<div style='margin: 10px 0; padding: 10px; border: 1px solid #ddd; background: #f9f9f9;'>";
        echo "<strong>Student ID:</strong> {$student_id}<br>";
        echo "<strong>GAT Predicted Level:</strong> {$level_pre_test}<br>";
        echo "<strong>Confidence:</strong> " . number_format($confidence, 4) . "<br>";
        echo "<strong>IRT Ability:</strong> " . number_format($student_result['features']['irt_ability'], 4) . "<br>";
        
        // Check apakah murid sudah dihitung pre_test (EXACT DARI FILE ASLI)
        $sql = "SELECT * FROM pre_test_result WHERE student_id = '{$student_id}'";
        $hasil = mysqli_query($conn, $sql);
        
        if (mysqli_num_rows($hasil) > 0) {
            echo "<span style='color: orange;'>⚠️ Sudah dihitung sebelumnya</span>";
        } else {
            // Simpan hasil pre_test ke database (EXACT DARI FILE ASLI)
            $sql = "INSERT INTO pre_test_result (student_id, level) VALUES('{$student_id}', '{$level_pre_test}')";
            $query = mysqli_query($conn, $sql);
            
            if (!$query) {
                echo "<span style='color: red;'>❌ Error: " . mysqli_error($conn) . "</span>";
                $error_count++;
            } else {
                // Ambil data survey dan hitung level final (EXACT DARI FILE ASLI)
                $sql = "SELECT level_result FROM survey_result WHERE student_id = '{$student_id}'";
                $query = mysqli_query($conn, $sql);
                $survey_data = mysqli_fetch_array($query, MYSQLI_ASSOC);
                $level_survey = $survey_data ? intval($survey_data['level_result']) : 3;
                
                // Level final = minimum dari pre-test dan survey (EXACT DARI FILE ASLI)
                $level_final = min($level_pre_test, $level_survey);
                
                // Simpan level final (EXACT DARI FILE ASLI)
                $sql = "INSERT INTO level_student (student_id, level) VALUES('{$student_id}', '{$level_final}') 
                        ON DUPLICATE KEY UPDATE level = '{$level_final}'";
                $query = mysqli_query($conn, $sql);
                
                if (!$query) {
                    echo "<span style='color: red;'>❌ Error saving final level: " . mysqli_error($conn) . "</span>";
                    $error_count++;
                } else {
                    echo "<span style='color: green;'>✅ Saved! Final Level: {$level_final} (min of GAT:{$level_pre_test}, Survey:{$level_survey})</span>";
                    $processed_count++;
                }
            }
        }
        echo "</div>";
    }
    
    // Show errors from API if any
    if (!empty($gat_response['data']['errors'])) {
        echo "<div style='color: red; border: 1px solid red; padding: 10px; margin: 10px 0;'>";
        echo "<h4>⚠️ API Errors:</h4>";
        foreach ($gat_response['data']['errors'] as $error) {
            echo "<p>Student {$error['student_id']}: {$error['error']}</p>";
        }
        echo "</div>";
    }
    
    // Summary
    echo "<div style='margin-top: 20px; padding: 15px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px;'>";
    echo "<h3>🎯 SUMMARY HASIL PROCESSING</h3>";
    echo "<p>✅ <strong>Successfully processed:</strong> {$processed_count} students</p>";
    echo "<p>❌ <strong>Errors:</strong> {$error_count} students</p>";
    echo "<p>🚀 <strong>Total GAT API calls:</strong> 1 batch request</p>";
    echo "<p>⏱️ <strong>Processing method:</strong> IRT Analysis + GAT Prediction + Database Storage</p>";
    echo "<h4>🎉 Pre-test processing dengan GAT API selesai!</h4>";
    echo "</div>";
    
    if ($processed_count > 0) {
        echo "<script>";
        echo "setTimeout(function(){";
        echo "  if(confirm('Processing selesai! Redirect ke halaman pre-test?')) {";
        echo "    window.location.href = '../admin/pre-test.php';";
        echo "  }";
        echo "}, 2000);";
        echo "</script>";
        echo "<p><em>Auto redirect in 2 seconds... (or click OK to redirect now)</em></p>";
    }
}
?>